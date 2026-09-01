"""GRPO trainer for the Memory Manager.

Implements the paper's Eq. (3)::

    J(theta) = E [ (1/G) * sum_i rho_theta^(i) * A_i - beta * D_KL(pi_theta || pi_ref) ]

with per-action group-normalized advantages ``A_i = (r_i - mean(r)) / std(r)`` and the per-token
policy ratio ``rho_theta^(i) = pi_theta(o_i, m'_i) / pi_old(o_i, m'_i)``.

We keep the implementation transparent (not TRL's GRPOTrainer) because Memory-R1's reward requires
running the Answer Agent on a memory bank that we own — no HuggingFace ``Dataset`` map trick can
represent that cleanly.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from memory_r1.prompts.manager import MEMORY_MANAGER_SYSTEM, build_manager_prompt
from memory_r1.training.config import TrainerConfig
from memory_r1.training.reward_pipeline import (
    ManagerTrainSample,
    score_manager_batch,
)
from memory_r1.training.rollout import build_chat_prompt
from memory_r1.utils import logger, resolve_attn_impl, resolve_device, resolve_dtype



@dataclass
class ManagerExample:
    """One training example loaded from the Algorithm 1 JSONL."""

    sample: ManagerTrainSample
    old_view: list[dict[str, str]]  # {"id","text"} shape to feed the prompt

    def render_user_prompt(self) -> str:
        return build_manager_prompt(
            retrieved_facts=self.sample.facts,
            old_memory=self.old_view,
        )


def load_manager_examples(path: Path, max_examples: int | None = None) -> list[ManagerExample]:
    """Load Algorithm-1 tuples; drop examples without linked QA (they carry no reward signal)."""

    out: list[ManagerExample] = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qas = rec.get("linked_qa") or []
            if not qas:
                continue
            qa = qas[0]  # any linked QA works; the paper's reward is per-QA.
            sample = ManagerTrainSample(
                speaker=rec["speaker"],
                facts=rec["extracted_facts"],
                old_bank_state=rec["temporal_memory_bank"],
                qa={"question": qa["question"], "answer": qa["answer"]},
                turn_timestamp=rec.get("timestamp"),
            )
            # Build the small ``old_memory`` view for the Manager: full per-speaker list from the
            # temporal bank. If it's huge we'd cap here, but 24-turn banks stay small.
            view = [
                {"id": e["id"], "text": e["text"]}
                for e in rec["temporal_memory_bank"]["entries"].get(rec["speaker"], [])
            ]
            out.append(ManagerExample(sample=sample, old_view=view))
            if max_examples is not None and len(out) >= max_examples:
                break
    return out


# --------------------------------------------------------------------------- log-prob helper


def _response_logprobs(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (per-token logprobs of ``response`` under ``model``, response_ids).

    We form ``prompt + response``, run one forward pass, and gather the log-probs at positions
    corresponding to the response tokens.
    """

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    with torch.set_grad_enabled(model.training):
        out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :]  # predict next token; last logit predicts nothing.
    target_positions = torch.arange(prompt_ids.shape[1] - 1, input_ids.shape[1] - 1, device=device)
    tgt_logits = logits[:, target_positions, :]
    targets = input_ids[:, prompt_ids.shape[1] :]
    logprobs = F.log_softmax(tgt_logits.float(), dim=-1)
    gathered = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return gathered[0], response_ids[0]


# --------------------------------------------------------------------------- GRPO trainer


class GRPOManagerTrainer:
    """Minimal group-relative PPO trainer for the Memory Manager.

    The trainer is deliberately small (< 300 lines) and works on a single GPU or CPU. For
    multi-GPU it's easy to swap the ``model`` for ``accelerate.prepare(model)``.
    """

    def __init__(
        self,
        cfg: TrainerConfig,
        reward_fn,  # (raw_str, ManagerTrainSample) -> float
        examples: list[ManagerExample],
    ) -> None:
        self.cfg = cfg
        self.reward_fn = reward_fn
        self.examples = examples

        logger.info("🧠 loading Memory Manager backbone: {}", cfg.model.name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name_or_path, trust_remote_code=cfg.model.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = resolve_device()
        dtype = resolve_dtype(cfg.model.dtype, self.device)
        attn_impl = resolve_attn_impl(self.device)
        logger.info("🖥️  device={} dtype={} attn_impl={}", self.device, dtype, attn_impl or "default")
        load_kwargs = dict(torch_dtype=dtype, trust_remote_code=cfg.model.trust_remote_code)
        if attn_impl is not None:
            load_kwargs["attn_implementation"] = attn_impl
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **load_kwargs)
        self.ref_model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **load_kwargs)
        for p in self.ref_model.parameters():
            p.requires_grad = False

        if cfg.model.use_peft:
            from peft import LoraConfig, get_peft_model

            lora = LoraConfig(
                r=cfg.model.peft_r,
                lora_alpha=cfg.model.peft_alpha,
                lora_dropout=cfg.model.peft_dropout,
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            self.model = get_peft_model(self.model, lora)

        self.model.to(self.device)
        self.ref_model.to(self.device)
        self.ref_model.eval()

        self.optim = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=cfg.optim.actor_lr, betas=(0.9, 0.95)
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optim, num_warmup_steps=10, num_training_steps=cfg.rl.total_steps
        )

    # ------------------------------------------------------------------ steps

    def _sample_group(self, prompt: str) -> list[str]:
        """Sample G candidates one at a time, each seeded independently.

        See ``grpo_answer.GRPOAnswerTrainer._sample_group`` for the rationale — batched
        ``num_return_sequences`` on MPS returns identical candidates.
        """

        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        gen_kwargs = dict(
            max_new_tokens=self.cfg.rl.max_response_length,
            temperature=self.cfg.rl.train_temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            repetition_penalty=self.cfg.rl.train_repetition_penalty,
        )
        if self.cfg.rl.train_top_p < 1.0:
            gen_kwargs["top_p"] = self.cfg.rl.train_top_p
        if self.cfg.rl.train_top_k > 0:
            gen_kwargs["top_k"] = self.cfg.rl.train_top_k

        # CUDA/CPU: batched sampling is G× faster than the loop.
        if str(self.device).startswith("cuda") or self.device == "cpu":
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs, num_return_sequences=self.cfg.rl.group_size, **gen_kwargs
                )
            completions = out[:, inputs["input_ids"].shape[-1] :]
            return [self.tokenizer.decode(c, skip_special_tokens=True).strip() for c in completions]

        # MPS: per-candidate loop with explicit seeding (batched num_return_sequences returns
        # identical candidates due to shared RNG state on MPS).
        from transformers import set_seed

        responses: list[str] = []
        base_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        for i in range(self.cfg.rl.group_size):
            seed_i = base_seed + i
            set_seed(seed_i)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed_i)
            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen_kwargs)
            completion = out[0, inputs["input_ids"].shape[-1] :]
            responses.append(self.tokenizer.decode(completion, skip_special_tokens=True).strip())
        return responses

    def _grpo_step(self, prompt: str, responses: list[str], rewards: list[float]) -> dict[str, float]:
        """Take one gradient step on Eq. (3)."""

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

        losses = []
        kls = []
        self.model.train()
        for resp, a in zip(responses, adv):
            new_lp, _ = _response_logprobs(self.model, self.tokenizer, prompt, resp, self.device)
            with torch.no_grad():
                old_lp, _ = _response_logprobs(self.ref_model, self.tokenizer, prompt, resp, self.device)
            ratio = torch.exp(new_lp - old_lp).clamp(max=10.0)
            # Clipped GRPO objective per Eq. (3), reduced to per-token then meaned.
            unclipped = ratio * a
            clipped = ratio.clamp(1 - self.cfg.optim.clip_range, 1 + self.cfg.optim.clip_range) * a
            pol = -torch.min(unclipped, clipped).mean()
            kl = (new_lp.exp() * (new_lp - old_lp)).mean()  # forward KL proxy
            losses.append(pol + self.cfg.rl.grpo_beta * kl)
            kls.append(kl.detach())

        loss = torch.stack(losses).mean()
        self.optim.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.max_grad_norm)
        self.optim.step()
        self.scheduler.step()

        return {
            "loss": float(loss.item()),
            "kl": float(torch.stack(kls).mean().item()),
            "grad_norm": float(grad),
            "reward_mean": float(rewards_t.mean().item()),
            "reward_std": float(rewards_t.std().item()),
        }

    # ------------------------------------------------------------------ loop

    def train(self) -> None:
        cfg = self.cfg
        out_dir = Path(cfg.output_dir) / cfg.run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("🏋️  GRPO Manager training: {} examples, {} steps, G={}",
                    len(self.examples), cfg.rl.total_steps, cfg.rl.group_size)
        example_iter = _cyclic(self.examples)

        for step in range(1, cfg.rl.total_steps + 1):
            t0 = time.time()
            examples_per_step = max(1, cfg.optim.total_batch_size // cfg.rl.group_size)
            batch: list[ManagerExample] = [next(example_iter) for _ in range(examples_per_step)]
            logger.info("🔄 step {}/{} — {} examples × G={} candidates",
                        step, cfg.rl.total_steps, examples_per_step, cfg.rl.group_size)

            step_stats: dict[str, float] = {}
            for ex_idx, ex in enumerate(batch):
                prompt = build_chat_prompt(self.tokenizer, MEMORY_MANAGER_SYSTEM, ex.render_user_prompt())
                logger.debug("💬 ex {}/{} speaker={} facts={}",
                             ex_idx + 1, len(batch), ex.sample.speaker, len(ex.sample.facts))

                t_roll = time.time()
                responses = self._sample_group(prompt)
                logger.debug("🎲 sampled {} candidates in {:.1f}s", len(responses), time.time() - t_roll)

                rewards = score_manager_batch(
                    self.reward_fn,
                    raws=responses,
                    samples=[ex.sample] * len(responses),
                    max_workers=1,
                )
                for ci, r in enumerate(rewards):
                    logger.debug("🎯 candidate {} reward={:.3f}", ci + 1, r)

                stats = self._grpo_step(prompt, responses, rewards)
                for k, v in stats.items():
                    step_stats.setdefault(k, 0.0)
                    step_stats[k] += v / len(batch)

            step_stats["step_time"] = time.time() - t0
            logger.info(
                "📉 step {}/{} loss={:.4f} reward_mean={:.3f} reward_std={:.3f} kl={:.4f} "
                "grad_norm={:.3f} dt={:.1f}s",
                step, cfg.rl.total_steps, step_stats["loss"], step_stats["reward_mean"],
                step_stats["reward_std"], step_stats["kl"], step_stats["grad_norm"],
                step_stats["step_time"],
            )
            (out_dir / "train_log.jsonl").open("a").write(
                json.dumps({"step": step, **step_stats}) + "\n"
            )
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
                logger.success("✅ checkpoint saved: {}", ckpt_dir)


def _cyclic(items):
    while True:
        for x in items:
            yield x
