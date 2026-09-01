"""Unified chat-LLM abstraction supporting OpenAI-compatible APIs and local HF models.

The paper uses:

- GPT-4o-mini for fact extraction (data construction) and LLM-as-a-Judge (evaluation).
- LLaMA-3.1-8B-Instruct / Qwen-2.5-{3,7,14}B-Instruct for the Memory Manager and Answer Agent.

We deliberately keep this abstraction thin: agents call ``LLMBackend.chat`` with a list of messages
and read back the string response. Concrete backends handle tokenization, sampling, and (for RL
training) the underlying policy model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMBackend(Protocol):
    """Minimal chat interface used by all agents."""

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> str: ...


# --------------------------------------------------------------------------- OpenAI backend


class OpenAIBackend:
    """Thin wrapper around ``openai.OpenAI`` for GPT-4o-mini and compatible endpoints.

    We support:

    - Default OpenAI (``OPENAI_API_KEY``).
    - Custom base URL via ``OPENAI_BASE_URL`` (useful for Azure OpenAI, together.ai, groq, etc.).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> str:
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(
            wait=wait_exponential(multiplier=1, min=1, max=30),
            stop=stop_after_attempt(6),
            reraise=True,
        )
        def _call() -> str:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            return (resp.choices[0].message.content or "").strip()

        return _call()


# --------------------------------------------------------------------------- HF backend


class HFBackend:
    """Local HuggingFace ``transformers`` chat backend.

    Loads a model + tokenizer once and reuses them for all ``chat`` calls. Intended for evaluation
    and quick manager/agent runs; the trainers use their own accelerated policies.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        repetition_penalty: float | None = None,
        attn_implementation: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        repetition_penalty : float | None
            Default = 1.0 on CUDA, 1.15 on MPS. The MPS softmax on long prompts with bfloat16
            occasionally collapses onto a single token (classic "!!!!!!" degeneration); a small
            repetition penalty breaks the loop without measurably hurting generation quality.
            Not needed on CUDA/A100/H100 where numerics are stable.
        attn_implementation : str | None
            Passed to ``AutoModelForCausalLM.from_pretrained``. Auto-selects
            ``"flash_attention_2"`` on CUDA + bf16 when ``flash-attn`` is installed (3-5×
            speedup on H100). Falls back to ``"sdpa"`` otherwise.
        """

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        from memory_r1.utils import resolve_device, resolve_dtype

        resolved_device = device or resolve_device()
        torch_dtype = resolve_dtype(dtype, resolved_device)

        # Device-appropriate defaults for generation-quality knobs.
        if repetition_penalty is None:
            repetition_penalty = 1.15 if resolved_device == "mps" else 1.0
        self.repetition_penalty = repetition_penalty

        # Try Flash Attention 2 on CUDA + bf16 when the wheel is available; big H100 speedup.
        if attn_implementation is None:
            if resolved_device == "cuda" and torch_dtype in (torch.bfloat16, torch.float16):
                try:
                    import flash_attn  # noqa: F401

                    attn_implementation = "flash_attention_2"
                except ImportError:
                    attn_implementation = "sdpa"
            else:
                attn_implementation = "sdpa"

        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "attn_implementation": attn_implementation,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if not load_in_4bit:
            self.model = self.model.to(resolved_device)
        self.model.eval()
        self.device = self.model.device

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> str:
        import torch

        chat = [{"role": m.role, "content": m.content} for m in messages]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        do_sample = temperature > 0
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-5),
                do_sample=do_sample,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        if stop:
            for s in stop:
                if s and s in text:
                    text = text.split(s)[0]
        return text
