"""PPO trainer for the Memory Manager (Eq. (2) of the paper).

We implement it via TRL's ``PPOTrainer`` where possible, falling back to a compact custom loop for
maximum compatibility across TRL versions. The reward is EM against the frozen Answer Agent's
output — same as GRPO — but PPO adds a critic (paper: ``critic_lr = 1e-5``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from memory_r1.prompts.manager import MEMORY_MANAGER_SYSTEM
from memory_r1.training.config import TrainerConfig
from memory_r1.training.grpo_manager import (
    ManagerExample,
    _response_logprobs,
    load_manager_examples,  # noqa: F401 (re-exported for external callers)
)
from memory_r1.training.reward_pipeline import score_manager_batch
from memory_r1.training.rollout import build_chat_prompt
from memory_r1.utils import logger, resolve_device, resolve_dtype



class ValueHead(torch.nn.Module):
    """One-hidden-layer MLP that maps last hidden state to a scalar value."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc(hidden).squeeze(-1)


class PPOManagerTrainer:
    """PPO-clip trainer for the Memory Manager.

    Same rollout / reward pipeline as GRPO but scalar-per-response advantages come from a critic
    trained with MSE against Monte-Carlo returns (undiscounted, since one full response = one
    trajectory in our setting).
    """

    def __init__(
        self,
        cfg: TrainerConfig,
        reward_fn,
        examples: list[ManagerExample],
    ) -> None:
        self.cfg = cfg
        self.reward_fn = reward_fn
        self.examples = examples

        logger.info("🧠 loading Memory Manager backbone (PPO): {}", cfg.model.name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name_or_path, trust_remote_code=cfg.model.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = resolve_device()
        dtype = resolve_dtype(cfg.model.dtype, self.device)
        logger.info("🖥️  device={} dtype={}", self.device, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name_or_path, torch_dtype=dtype, trust_remote_code=cfg.model.trust_remote_code
        )
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name_or_path, torch_dtype=dtype, trust_remote_code=cfg.model.trust_remote_code
        )
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.value_head = ValueHead(self.model.config.hidden_size)

        self.model.to(self.device)
        self.ref_model.to(self.device).eval()
        self.value_head.to(self.device).to(dtype)

        self.optim_actor = torch.optim.AdamW(self.model.parameters(), lr=cfg.optim.actor_lr, betas=(0.9, 0.95))
        self.optim_critic = torch.optim.AdamW(self.value_head.parameters(), lr=cfg.optim.critic_lr)
        self.sched_actor = get_cosine_schedule_with_warmup(
            self.optim_actor, num_warmup_steps=10, num_training_steps=cfg.rl.total_steps
        )
        self.sched_critic = get_cosine_schedule_with_warmup(
            self.optim_critic, num_warmup_steps=10, num_training_steps=cfg.rl.total_steps
        )

    # ------------------------------------------------------------------ helpers

    def _value_from_response(self, prompt: str, response: str) -> torch.Tensor:
        input_ids = self.tokenizer(prompt + response, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        out = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        last_hidden = out.hidden_states[-1][:, -1, :]  # (1, H)
        return self.value_head(last_hidden).squeeze(0)  # scalar

    def _sample_one(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.rl.max_response_length,
                temperature=self.cfg.rl.train_temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()

    def _ppo_step(self, prompt: str, response: str, reward: float) -> dict[str, float]:
        self.model.train()

        new_lp, _ = _response_logprobs(self.model, self.tokenizer, prompt, response, self.device)
        with torch.no_grad():
            old_lp, _ = _response_logprobs(self.ref_model, self.tokenizer, prompt, response, self.device)
        ratio = torch.exp(new_lp - old_lp).clamp(max=10.0)

        value = self._value_from_response(prompt, response).float()
        adv = torch.tensor(reward, dtype=value.dtype, device=self.device) - value.detach()
        returns = torch.tensor(reward, dtype=value.dtype, device=self.device)

        unclipped = ratio * adv
        clipped = ratio.clamp(1 - self.cfg.optim.clip_range, 1 + self.cfg.optim.clip_range) * adv
        policy_loss = -torch.min(unclipped, clipped).mean()

        kl = (new_lp.exp() * (new_lp - old_lp)).mean()
        loss_actor = policy_loss + self.cfg.optim.kl_coef * kl

        self.optim_actor.zero_grad()
        loss_actor.backward()
        grad_a = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.max_grad_norm)
        self.optim_actor.step()
        self.sched_actor.step()

        loss_critic = F.mse_loss(value, returns)
        self.optim_critic.zero_grad()
        loss_critic.backward()
        grad_c = torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.cfg.optim.max_grad_norm)
        self.optim_critic.step()
        self.sched_critic.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(loss_critic.item()),
            "reward": reward,
            "kl": float(kl.item()),
            "grad_norm_actor": float(grad_a),
            "grad_norm_critic": float(grad_c),
        }

    # ------------------------------------------------------------------ loop

    def train(self) -> None:
        cfg = self.cfg
        out_dir = Path(cfg.output_dir) / cfg.run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        from memory_r1.training.grpo_manager import _cyclic

        example_iter = _cyclic(self.examples)
        for step in range(1, cfg.rl.total_steps + 1):
            t0 = time.time()
            batch_examples = [next(example_iter) for _ in range(cfg.optim.total_batch_size // 8)]
            step_stats: dict[str, float] = {}
            for ex in batch_examples:
                prompt = build_chat_prompt(self.tokenizer, MEMORY_MANAGER_SYSTEM, ex.render_user_prompt())
                response = self._sample_one(prompt)
                rewards = score_manager_batch(self.reward_fn, [response], [ex.sample], max_workers=1)
                s = self._ppo_step(prompt, response, rewards[0])
                for k, v in s.items():
                    step_stats.setdefault(k, 0.0)
                    step_stats[k] += v / len(batch_examples)

            step_stats["step_time"] = time.time() - t0
            logger.info(
                "📉 step {}/{} policy_loss={:.4f} value_loss={:.4f} reward={:.3f} kl={:.4f} dt={:.1f}s",
                step, cfg.rl.total_steps, step_stats["policy_loss"], step_stats["value_loss"],
                step_stats["reward"], step_stats["kl"], step_stats["step_time"],
            )
            (out_dir / "train_log.jsonl").open("a").write(json.dumps({"step": step, **step_stats}) + "\n")
            (out_dir / "status.json").write_text(json.dumps({
                "run_name": cfg.run_name,
                "step": step,
                "total_steps": cfg.rl.total_steps,
                "progress": step / cfg.rl.total_steps,
                **step_stats,
            }, indent=2))

            if step % cfg.rl.save_every == 0 or step == cfg.rl.total_steps:
                ckpt_dir = out_dir / f"step_{step}"
                logger.info("💾 saving checkpoint → {}", ckpt_dir)
                self.model.save_pretrained(ckpt_dir)
                self.tokenizer.save_pretrained(ckpt_dir)
                torch.save(self.value_head.state_dict(), ckpt_dir / "value_head.pt")
                logger.success("✅ checkpoint saved: {}", ckpt_dir)
