"""Verify that the local Qwen2.5-7B-Instruct checkpoint loads and generates coherently.

Usage:
    python scripts/verify_qwen2_5_7b.py
    python scripts/verify_qwen2_5_7b.py --model-dir /custom/path/to/qwen2.5-7b

Passes if:
    - Config declares ``model_type=qwen2`` with the expected 28-layer / 3584-hidden shape
    - Tokenizer + model load on the resolved device with the resolved dtype
    - A short chat-templated generation returns non-empty text with no ``nan``/``inf`` logits

Qwen2.5-7B-Instruct is the second Table-1 backbone in the paper (not-gated, easier
to grab than LLaMA). Fits comfortably in bf16 on an 80 GB H100 (~14 GB weights).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "qwen2.5-7b"
)
EXPECTED_MODEL_TYPE = "qwen2"
EXPECTED_HIDDEN_SIZE = 3584
EXPECTED_NUM_LAYERS = 28

# Same LoCoMo-flavored prompts as the sibling verify_* scripts so a regression in one
# is likely to surface in the others too.
SHORT_PROMPT = "In one short sentence, what is retrieval-augmented generation?"
LONG_PROMPT_QUESTION = "When did Caroline go to the LGBTQ support group?"
LONG_PROMPT_CONTEXT = (
    "You have the following notes from a conversation. Use them to answer the "
    "question at the end.\n\n"
    + "\n".join(
        f"- Note {i:02d}: On day {i}, Caroline mentioned attending various support "
        f"meetings — book club on Mondays, running group on Tuesdays, and cooking "
        f"class on Wednesdays. She skipped the Thursday events because of work."
        for i in range(1, 41)
    )
    + "\n- Note 41: On 7 May 2023, Caroline went to the LGBTQ support group for the first time.\n"
    + "\n".join(
        f"- Note {i:02d}: Later that week she also visited the gym and a coffee shop."
        for i in range(42, 81)
    )
    + f"\n\nQuestion: {LONG_PROMPT_QUESTION}\nAnswer briefly."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Local model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument("--device", default=None, help="cpu / cuda / mps (default: auto)")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Tokens to generate per prompt (default: 64)",
    )
    args = parser.parse_args()

    model_dir: Path = args.model_dir.resolve()
    print(f"→ model dir: {model_dir}")

    if not model_dir.is_dir():
        print(f"❌ not a directory: {model_dir}")
        return 1
    if not (model_dir / "config.json").exists():
        print(f"❌ missing config.json in {model_dir}")
        return 1

    try:
        import json

        cfg = json.loads((model_dir / "config.json").read_text())
    except Exception as e:
        print(f"❌ failed to parse config.json: {type(e).__name__}: {e}")
        return 1

    print(f"→ config: model_type={cfg.get('model_type')!r}, "
          f"hidden_size={cfg.get('hidden_size')}, num_hidden_layers={cfg.get('num_hidden_layers')}")
    if cfg.get("model_type") != EXPECTED_MODEL_TYPE:
        print(f"❌ expected model_type={EXPECTED_MODEL_TYPE!r}, got {cfg.get('model_type')!r}")
        return 1
    if cfg.get("hidden_size") != EXPECTED_HIDDEN_SIZE:
        print(f"❌ expected hidden_size={EXPECTED_HIDDEN_SIZE}, got {cfg.get('hidden_size')}")
        return 1
    if cfg.get("num_hidden_layers") != EXPECTED_NUM_LAYERS:
        print(f"❌ expected num_hidden_layers={EXPECTED_NUM_LAYERS}, "
              f"got {cfg.get('num_hidden_layers')}")
        return 1

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"❌ transformers / torch not installed: {e}")
        return 1

    # Standalone helpers (inlined so this diagnostic works pre `pip install -e .`).
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(name: str, dev: str) -> "torch.dtype":
        m = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dt = m.get(name, torch.bfloat16)
        if dev == "cpu" and dt == torch.bfloat16:
            return torch.float32  # bf16 on CPU is ~10× slower than fp32
        return dt

    def _resolve_attn_impl(dev: str) -> "str | None":
        # MPS SDPA silently NaNs during long bf16 decode → force eager on MPS.
        return "eager" if dev == "mps" else None

    device = args.device or _resolve_device()
    dtype = _resolve_dtype(cfg.get("torch_dtype", "bfloat16"), device)
    attn_impl = _resolve_attn_impl(device)
    print(f"→ device={device}, dtype={dtype}, attn_impl={attn_impl or 'default'}")

    print("→ loading tokenizer …")
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir))
    except Exception as e:
        print(f"❌ tokenizer load failed: {type(e).__name__}: {e}")
        return 1
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("→ loading model …")
    load_kwargs = dict(torch_dtype=dtype)
    if attn_impl is not None:
        load_kwargs["attn_implementation"] = attn_impl
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs).to(device).eval()
    except Exception as e:
        print(f"❌ model load failed: {type(e).__name__}: {e}")
        return 1
    print(f"✓ loaded — dtype={next(model.parameters()).dtype}, device={next(model.parameters()).device}")

    for label, user_msg in [("short prompt", SHORT_PROMPT), ("long prompt", LONG_PROMPT_CONTEXT)]:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
        n_tok = len(tok.encode(prompt))
        print(f"\n→ generating ({label}, {n_tok} prompt tokens, "
              f"max_new_tokens={args.max_new_tokens}) …")
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
        finite = bool(torch.isfinite(logits).all().item())
        print(f"   logits: dtype={logits.dtype} finite={finite} max_abs={logits.abs().max().item():.2f}")
        if not finite:
            print("❌ non-finite logits — generation would fail.")
            return 1

        try:
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
        except RuntimeError as e:
            print(f"❌ generate() failed: {e}")
            return 1
        gen = tok.decode(out[0, inputs.input_ids.shape[-1] :], skip_special_tokens=True).strip()
        print(f"   response ({len(gen)} chars): {gen[:200]!r}")
        if not gen:
            print("❌ empty generation")
            return 1

    print("\n✅ verification passed — Qwen2.5-7B loads, config matches, generation is numerically stable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
