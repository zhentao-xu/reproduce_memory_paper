"""Shared rollout utilities: build prompts + sample from the policy.

Both the Manager and Answer trainers share the same rollout loop: given a batch of chat prompts
(system + user), sample the policy N times per prompt, compute per-response rewards, then feed
those into TRL's PPO / GRPO objective.

We keep the loop lightweight and compatible with both models — the trainer instantiates a
``PolicyRollout`` that owns the tokenizer + generation config.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from memory_r1.utils import resolve_device


@dataclass
class RolloutSample:
    """A single (prompt, sampled_response, reward) triple used by the RL trainer."""

    prompt_ids: torch.Tensor  # (P,)
    response_ids: torch.Tensor  # (R,)
    response_text: str
    reward: float


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    """Render a chat template into a string suitable for tokenizer(...) directly.

    Works for LLaMA-3.1-Instruct and Qwen2.5-Instruct (both ship chat templates).
    """

    chat = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def sample_responses(
    model,
    tokenizer,
    prompts: list[str],
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    device: str | torch.device | None = None,
) -> list[list[str]]:
    """Sample ``group_size`` responses per prompt. Returns [[str, ...], ...] parallel to ``prompts``.

    Used by GRPO (group of G candidates) and PPO (group_size=1).
    """

    if device is None:
        device = resolve_device()

    all_responses: list[list[str]] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                do_sample=True,
                num_return_sequences=group_size,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completions = gen[:, inputs["input_ids"].shape[-1] :]
        texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
        all_responses.append([t.strip() for t in texts])
    return all_responses
