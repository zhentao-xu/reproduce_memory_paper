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
    ) -> None:
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
        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype, "trust_remote_code": trust_remote_code}
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
            # ``repetition_penalty=1.15`` prevents the classic "!!!!!!!!!!" degeneration when the
            # softmax collapses onto a low-info token under bfloat16 on MPS with long prompts.
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-5),
                do_sample=do_sample,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        if stop:
            for s in stop:
                if s and s in text:
                    text = text.split(s)[0]
        return text
