"""GRPO trainer for the Answer Agent.

Same objective as the Memory Manager's GRPO trainer (Eq. (3)); the difference is the reward, which
here is direct EM against the gold answer for the (question, retrieved_60) input.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from memory_r1.prompts.answer import ANSWER_AGENT_SYSTEM, build_answer_prompt
from memory_r1.training.config import TrainerConfig
from memory_r1.training.grpo_manager import _cyclic, _response_logprobs
from memory_r1.training.reward_pipeline import AnswerTrainSample, score_answer_batch
from memory_r1.training.rollout import build_chat_prompt
from memory_r1.utils import logger, resolve_device, resolve_dtype



@dataclass
class AnswerExample:
    sample: AnswerTrainSample

    def render_user_prompt(self) -> str:
        return build_answer_prompt(self.sample.question, self.sample.retrieved)


def load_answer_examples(path: Path, max_examples: int | None = None) -> list[AnswerExample]:
    """Load Algorithm-2 tuples."""

    out: list[AnswerExample] = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sample = AnswerTrainSample(
                question=rec["question"],
                gold_answer=rec["gold_answer"],
                retrieved=rec["retrieved"],
            )
            out.append(AnswerExample(sample=sample))
            if max_examples is not None and len(out) >= max_examples:
                break
    return out


class GRPOAnswerTrainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        reward_fn,
        examples: list[AnswerExample],
    ) -> None:
        self.cfg = cfg
        self.reward_fn = reward_fn
        self.examples = examples

        logger.info("Loading Answer Agent backbone: {}", cfg.model.name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name_or_path, trust_remote_code=cfg.model.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = resolve_device()
        dtype = resolve_dtype(cfg.model.dtype, self.device)
        logger.info("Device: {}, dtype: {}", self.device, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name_or_path, torch_dtype=dtype, trust_remote_code=cfg.model.trust_remote_code
        )
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name_or_path, torch_dtype=dtype, trust_remote_code=cfg.model.trust_remote_code
        )
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
        self.ref_model.to(self.device).eval()

        self.optim = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=cfg.optim.actor_lr, betas=(0.9, 0.95)
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optim, num_warmup_steps=10, num_training_steps=cfg.rl.total_steps
        )

    def _sample_group(self, prompt: str) -> list[str]:
        """Sample G candidates one at a time, each seeded independently.

        Batched ``num_return_sequences=G`` produces IDENTICAL candidates on MPS (torch RNG state is
        shared across the parallel decode paths). We loop instead — slower by a factor of G, but
        candidates are actually independent draws from the sampling distribution, so GRPO's
        group-relative advantage has non-zero variance to work with.
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

        responses: list[str] = []
        # Reseed *every* torch RNG for each candidate. MPS has its own RNG that
        # ``torch.manual_seed`` does NOT touch — we need ``torch.mps.manual_seed`` explicitly.
        # Transformers also caches sampling state via ``set_seed``.
        from transformers import set_seed

        base_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        for i in range(self.cfg.rl.group_size):
            seed_i = base_seed + i
            set_seed(seed_i)  # sets python, numpy, torch (all backends)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed_i)
            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen_kwargs)
            completion = out[0, inputs["input_ids"].shape[-1] :]
            responses.append(self.tokenizer.decode(completion, skip_special_tokens=True).strip())
        return responses

    def _step(self, prompt: str, responses: list[str], rewards: list[float]) -> dict[str, float]:
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)
        losses, kls = [], []
        self.model.train()
        for resp, a in zip(responses, adv, strict=True):
            new_lp, _ = _response_logprobs(self.model, self.tokenizer, prompt, resp, self.device)
            with torch.no_grad():
                old_lp, _ = _response_logprobs(self.ref_model, self.tokenizer, prompt, resp, self.device)
            ratio = torch.exp(new_lp - old_lp).clamp(max=10.0)
            unclipped = ratio * a
            clipped = ratio.clamp(1 - self.cfg.optim.clip_range, 1 + self.cfg.optim.clip_range) * a
            pol = -torch.min(unclipped, clipped).mean()
            kl = (new_lp.exp() * (new_lp - old_lp)).mean()
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

    def train(self) -> None:
        cfg = self.cfg
        out_dir = Path(cfg.output_dir) / cfg.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("🏋️  starting training: total_steps={} batch={} group={}",
                    cfg.rl.total_steps, cfg.optim.total_batch_size, cfg.rl.group_size)

        it = _cyclic(self.examples)
        for step in range(1, cfg.rl.total_steps + 1):
            t0 = time.time()
            examples_per_step = max(1, cfg.optim.total_batch_size // cfg.rl.group_size)
            batch = [next(it) for _ in range(examples_per_step)]
            logger.info("🔄 step {}/{} — {} examples × G={} candidates",
                        step, cfg.rl.total_steps, examples_per_step, cfg.rl.group_size)

            step_stats: dict[str, float] = {}
            for ex_idx, ex in enumerate(batch):
                prompt = build_chat_prompt(self.tokenizer, ANSWER_AGENT_SYSTEM, ex.render_user_prompt())
                logger.info("💬 ex {}/{} prompt_tokens={} question={!r}",
                            ex_idx + 1, len(batch),
                            len(self.tokenizer.encode(prompt)),
                            ex.sample.question[:120])
                logger.info("🥇 gold={!r}", ex.sample.gold_answer)

                # Dump the retrieved memories that will be shown to the model, so operators can
                # see whether the answer is inferrable from the retrieved context at all.
                for speaker, mems in ex.sample.retrieved.items():
                    for mi, mem in enumerate(mems):
                        ts = mem.get("timestamp", "")
                        text = mem.get("text", "")
                        ts_str = f"{ts}: " if ts else ""
                        logger.debug("🗃️  [{}] {}#{}: {}{}", speaker, speaker, mi + 1, ts_str, text[:200])

                # Dump the FULL prompt (system + user, after chat_template rendering) so the user
                # can copy-paste it into ChatGPT / Ollama / anywhere and reproduce the input we're
                # actually sending to Qwen3. INFO-level because reproducibility matters.
                logger.info("📜 FULL PROMPT BEGIN ===================================================")
                for line in prompt.splitlines():
                    logger.info("📜 | {}", line)
                logger.info("📜 FULL PROMPT END =====================================================")

                t_roll = time.time()
                responses = self._sample_group(prompt)
                logger.debug("🎲 sampled {} candidates in {:.1f}s", len(responses), time.time() - t_roll)

                rewards = score_answer_batch(
                    self.reward_fn, raws=responses, samples=[ex.sample] * len(responses), max_workers=1
                )
                for ci, (resp, r) in enumerate(zip(responses, rewards, strict=True)):
                    logger.debug("🎯 candidate {} reward={:.3f} len={} pred={!r}",
                                 ci + 1, r, len(resp), resp.replace("\n", " "))

                s = self._step(prompt, responses, rewards)
                for k, v in s.items():
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

            # Structured metrics dump.
            (out_dir / "train_log.jsonl").open("a").write(
                json.dumps({"step": step, **step_stats}) + "\n"
            )
            # ``status.json`` at fixed path so operators can ``cat outputs/.../status.json`` any time.
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
