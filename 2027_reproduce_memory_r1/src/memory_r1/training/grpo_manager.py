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


def _response_logprobs_batch(
    model,
    tokenizer,
    prompt: str,
    responses: list[str],
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized ``_response_logprobs`` for ``len(responses)`` responses sharing one prompt.

    All GRPO candidates in a group share the same prompt, so we tokenize the prompt once, tokenize
    each response independently, right-pad the response side to a common length, and run ONE batched
    forward. This trades a bit of wasted compute on padding tokens for cutting kernel launches ~B×.
    On H100 with a 2 k-token prompt + short responses, batching 4-8 candidates is a ~3× speedup vs.
    the per-candidate loop.

    Returns
    -------
    logprobs : Tensor  (B, T_r)
        Per-token log-probs of each response under ``model``. Positions past a candidate's true
        response length are junk — mask them with ``mask``.
    mask : Tensor  (B, T_r)
        1.0 where the token belongs to the real response, 0.0 for right-padding. Use this to
        compute per-candidate means / KL correctly.
    """

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    P = int(prompt_ids.shape[0])

    resp_id_list = [
        tokenizer(r, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
        for r in responses
    ]
    T_r = max(int(x.shape[0]) for x in resp_id_list)
    B = len(responses)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((B, P + T_r), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((B, P + T_r), dtype=torch.long, device=device)
    resp_mask = torch.zeros((B, T_r), dtype=torch.float32, device=device)
    for i, r_ids in enumerate(resp_id_list):
        L = int(r_ids.shape[0])
        input_ids[i, :P] = prompt_ids
        input_ids[i, P : P + L] = r_ids
        attn_mask[i, : P + L] = 1
        resp_mask[i, :L] = 1.0

    with torch.set_grad_enabled(model.training):
        out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    # Response tokens live at positions [P, P+T_r). Their predicting-logits are at [P-1, P+T_r-1).
    logits = out.logits[:, P - 1 : P - 1 + T_r, :]  # (B, T_r, V)
    targets = input_ids[:, P : P + T_r]  # (B, T_r)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    gathered = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T_r)
    return gathered, resp_mask


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

        # See grpo_answer.py for the rationale: with LoRA, π_ref == actor with adapter disabled,
        # so we skip the second model copy entirely and save ~8 GB (4B) / ~16 GB (8B) of VRAM.
        if cfg.model.use_peft:
            self.ref_model = None
            logger.info("♻️  reference model = adapter-disabled actor (saves ~{} GB VRAM)",
                        int(sum(p.numel() for p in self.model.parameters()) * 2 / 1e9))
        else:
            self.ref_model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **load_kwargs)
            for p in self.ref_model.parameters():
                p.requires_grad = False

        self.model.to(self.device)
        if self.ref_model is not None:
            self.ref_model.to(self.device).eval()

        # Gradient checkpointing is now a config knob (see grpo_answer.py). Default off since
        # LoRA + G=16 sits at ~30 % of an 80 GB H100 without checkpointing.
        if cfg.model.use_gradient_checkpointing:
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()
            logger.info("🪶 gradient checkpointing enabled on actor")

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

    def _ref_forward_batch(self, prompt: str, responses: list[str]):
        """Reference-policy log-probs — auto-routes to disable_adapter() when ref_model is None."""
        if self.ref_model is None:
            with self.model.disable_adapter():
                return _response_logprobs_batch(
                    self.model, self.tokenizer, prompt, responses, self.device
                )
        return _response_logprobs_batch(
            self.ref_model, self.tokenizer, prompt, responses, self.device
        )

    def _grpo_step(self, prompt: str, responses: list[str], rewards: list[float]) -> dict[str, float]:
        """Take one gradient step on Eq. (3).

        Microbatched candidates: ``optim.micro_batch_size_per_gpu`` candidates share one forward
        pass on the actor + reference, then backward. Peak activation memory scales with µ, not G.
        Loss inside each microbatch is scaled by 1/G so summed grads across all microbatches equal
        the mean-over-G loss the naive stacked-loss implementation would produce.
        """

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)
        G = len(responses)
        micro = max(1, min(G, int(self.cfg.optim.micro_batch_size_per_gpu)))

        self.model.train()
        self.optim.zero_grad()
        loss_sum, kl_sum = 0.0, 0.0
        for start in range(0, G, micro):
            end = min(G, start + micro)
            batch_resps = responses[start:end]
            batch_adv = adv[start:end].unsqueeze(1)

            new_lp, resp_mask = _response_logprobs_batch(
                self.model, self.tokenizer, prompt, batch_resps, self.device
            )
            with torch.no_grad():
                old_lp, _ = self._ref_forward_batch(prompt, batch_resps)

            ratio = torch.exp(new_lp - old_lp).clamp(max=10.0)
            unclipped = ratio * batch_adv
            clipped = ratio.clamp(1 - self.cfg.optim.clip_range, 1 + self.cfg.optim.clip_range) * batch_adv
            pol_per_tok = -torch.min(unclipped, clipped) * resp_mask
            kl_per_tok = (new_lp.exp() * (new_lp - old_lp)) * resp_mask
            tok_counts = resp_mask.sum(dim=1).clamp(min=1)
            pol = pol_per_tok.sum(dim=1) / tok_counts
            kl = kl_per_tok.sum(dim=1) / tok_counts

            loss_i = ((pol + self.cfg.rl.grpo_beta * kl) / G).sum()
            loss_i.backward()
            loss_sum += float(loss_i.item()) * G
            kl_sum += float(kl.sum().item())

        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.max_grad_norm)
        self.optim.step()
        self.scheduler.step()

        return {
            "loss": loss_sum / G,
            "kl": kl_sum / G,
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

                # Release sampling KV-cache blocks so _grpo_step can allocate its own
                # activation buffers contiguously. See grpo_answer.py for the full rationale.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
