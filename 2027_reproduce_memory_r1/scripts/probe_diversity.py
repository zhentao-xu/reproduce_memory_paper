"""Standalone sanity check: does Qwen3-4B actually produce DIFFERENT samples on MPS?

Loads the base model, tokenizes a short prompt, and calls ``generate()`` 5 times with different
seeds. Prints the raw token ids and the decoded text. If all 5 outputs are identical, sampling is
broken (either ``do_sample`` isn't being honored or MPS RNG isn't affecting the sampler).

Usage::

    MEMORY_R1_DEVICE=mps HF_HOME=./models uv run --no-sync python scripts/probe_diversity.py
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from memory_r1.utils import init_run_logger, logger


def main() -> None:
    init_run_logger("probe_diversity")

    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("⚙️  model={} device={}", model_name, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    prompt = "Tell me a short joke about programming."
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

    # Try 5 temperatures × 3 seeds, print outputs
    for temp in [0.7, 1.0, 1.5, 2.0]:
        logger.info("🌡️  temperature = {}", temp)
        for seed in [1, 2, 3]:
            set_seed(seed)
            if device == "mps":
                torch.mps.manual_seed(seed)
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    temperature=temp,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            completion_ids = out[0, inputs["input_ids"].shape[-1] :]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            first_5_ids = completion_ids[:5].tolist()
            logger.info("  seed={} first_ids={} text={!r}", seed, first_5_ids, completion_text[:120])


if __name__ == "__main__":
    main()
