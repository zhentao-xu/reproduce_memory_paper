"""Quick end-to-end sanity check: load an Answer Agent (base or fine-tuned) and answer a few
questions from the pre-built ``answer_train.jsonl``.

Bypasses the full 300-turn per-dialogue Memory Manager loop by using the retrieved-60-memory
snapshot we already computed via Algorithm 2. O(seconds per question) — perfect for a smoke
test that proves *training → inference → metrics* is wired up.

Usage::

    # Base model:
    MEMORY_R1_DEVICE=mps uv run python scripts/quick_eval_answer.py \\
        --model Qwen/Qwen3-4B-Instruct-2507 --n 3

    # Fine-tuned LoRA checkpoint:
    MEMORY_R1_DEVICE=mps uv run python scripts/quick_eval_answer.py \\
        --model outputs/checkpoints/smoke_qwen3_4b/step_1 \\
        --base Qwen/Qwen3-4B-Instruct-2507 --n 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_r1.agents.answer_agent import AnswerAgent
from memory_r1.agents.llm_backend import HFBackend
from memory_r1.eval.metrics import bleu1, exact_match, token_f1
from memory_r1.utils import init_run_logger, logger


def load_backend(model: str, base: str | None, dtype: str) -> HFBackend:
    if base is None:
        logger.info("📥 loading model: {}", model)
        return HFBackend(model_name_or_path=model, dtype=dtype)

    logger.info("📥 loading base: {}", base)
    backend = HFBackend(model_name_or_path=base, dtype=dtype)
    from peft import PeftModel

    logger.info("📥 loading LoRA adapter: {}", model)
    backend.model = PeftModel.from_pretrained(backend.model, model)
    backend.model.eval()
    return backend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or checkpoint path.")
    ap.add_argument("--base", default=None, help="Base model when --model is a LoRA adapter.")
    ap.add_argument("--data", default=Path("data/processed/answer_train.jsonl"), type=Path)
    ap.add_argument("--n", default=3, type=int, help="How many examples to evaluate.")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", default=128, type=int)
    args = ap.parse_args()

    init_run_logger("quick_eval_answer")
    logger.info("⚙️  model={} base={} n={}", args.model, args.base, args.n)

    backend = load_backend(args.model, args.base, args.dtype)
    agent = AnswerAgent(backend=backend, temperature=0.0, max_tokens=args.max_new_tokens)

    with args.data.open() as f:
        rows = [json.loads(line) for _, line in zip(range(args.n), f, strict=False)]
    logger.info("📚 loaded {} rows from {}", len(rows), args.data)

    results = []
    for i, r in enumerate(rows):
        question = r["question"]
        gold = r["gold_answer"]
        retrieved = r["retrieved"]
        logger.info("💬 Q{}/{}: {}", i + 1, len(rows), question)

        out = agent.answer(question=question, memories_by_speaker=retrieved)
        f1 = token_f1(out.answer, gold)
        b1 = bleu1(out.answer, gold)
        em = exact_match(out.answer, gold)
        results.append({"question": question, "gold": gold, "pred": out.answer, "f1": f1, "b1": b1, "em": em})
        logger.info("📝 gold={!r} pred={!r}", gold, out.answer)
        logger.info("📈 F1={:.2f} B1={:.2f} EM={:.2f}", f1, b1, em)

    if results:
        avg_f1 = sum(r["f1"] for r in results) / len(results)
        avg_b1 = sum(r["b1"] for r in results) / len(results)
        avg_em = sum(r["em"] for r in results) / len(results)
        logger.success(
            "🏁 average over n={}: F1={:.2f} B1={:.2f} EM={:.2f}",
            len(results), avg_f1 * 100, avg_b1 * 100, avg_em * 100,
        )


if __name__ == "__main__":
    main()
