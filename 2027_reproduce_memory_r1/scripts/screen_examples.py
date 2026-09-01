"""Screen training examples by base-model F1 to find "learnable" ones.

For GRPO to produce a non-zero gradient, the group of G candidates on a given example needs
reward *variance*. If the base model always answers correctly (F1≈1) or always wrong (F1≈0), the
group will collapse to a single reward value → advantage = 0 → no learning.

This script:

1. Runs the base Answer Agent on the first ``--n`` rows of ``answer_train.jsonl``.
2. Keeps rows where ``lo < F1 < hi`` (default 0.05 .. 0.85).
3. Writes them to ``--out`` for RL training.

Usage::

    MEMORY_R1_DEVICE=mps uv run --no-sync python scripts/screen_examples.py \\
        --model Qwen/Qwen3-4B-Instruct-2507 --n 12 \\
        --out data/processed/answer_train_filtered.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_r1.agents.answer_agent import AnswerAgent
from memory_r1.agents.llm_backend import HFBackend
from memory_r1.eval.metrics import bleu1, exact_match, token_f1
from memory_r1.utils import init_run_logger, logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", default=Path("data/processed/answer_train.jsonl"), type=Path)
    ap.add_argument("--out", default=Path("data/processed/answer_train_filtered.jsonl"), type=Path)
    ap.add_argument("--n", default=12, type=int, help="How many examples to probe.")
    ap.add_argument("--lo", default=0.05, type=float, help="Lower F1 bound (exclusive).")
    ap.add_argument("--hi", default=0.85, type=float, help="Upper F1 bound (exclusive).")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", default=96, type=int)
    ap.add_argument(
        "--truncate-memories-per-speaker",
        default=30,  # paper's default; drop to 5-10 on MPS to avoid bfloat16 degeneration.
        type=int,
        help="Cap retrieved memories per speaker at inference time. Paper uses 30 (=60 total). "
             "On MPS + bfloat16, drop to 5-10 to avoid prompt-length degeneration into '!!!!!!'.",
    )
    args = ap.parse_args()

    init_run_logger("screen_examples")
    logger.info("⚙️  probing first {} rows of {}", args.n, args.data)
    logger.info("⚙️  keeping rows with {:.2f} < F1 < {:.2f}", args.lo, args.hi)

    backend = HFBackend(model_name_or_path=args.model, dtype=args.dtype)
    agent = AnswerAgent(backend=backend, temperature=0.0, max_tokens=args.max_new_tokens)

    with args.data.open() as f:
        rows = [json.loads(line) for _, line in zip(range(args.n), f, strict=False)]
    logger.info("📚 loaded {} rows to probe", len(rows))

    kept: list[dict] = []
    all_scores: list[tuple[int, str, str, str, float, float, float]] = []
    for i, r in enumerate(rows):
        question, gold, retrieved = r["question"], r["gold_answer"], r["retrieved"]
        # Truncate retrieved to top-K per speaker at inference time — the answer_train.jsonl file
        # was built with paper's top-30 per speaker (60 total), which overflows Qwen3-4B on MPS
        # with bfloat16. 15 per speaker (30 total) keeps the prompt manageable.
        retrieved_capped = {sp: mems[: args.truncate_memories_per_speaker] for sp, mems in retrieved.items()}
        logger.info("💬 [{}/{}] {}", i + 1, len(rows), question[:80])
        out = agent.answer(question=question, memories_by_speaker=retrieved_capped)
        f1 = token_f1(out.answer, gold)
        b1 = bleu1(out.answer, gold)
        em = exact_match(out.answer, gold)
        logger.info("📝 gold={!r} pred={!r}", gold, out.answer)
        logger.info("📈 F1={:.2f} B1={:.2f} EM={:.2f}", f1, b1, em)
        all_scores.append((i + 1, question, gold, out.answer, f1, b1, em))
        if args.lo < f1 < args.hi:
            # Persist the truncated retrieved set (not the original 60) so downstream training
            # doesn't overflow the model's stable-context length.
            r_capped = dict(r)
            r_capped["retrieved"] = retrieved_capped
            kept.append(r_capped)
            logger.success("✅ kept row {} (F1 in learnable window)", i + 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.success("🏁 wrote {} learnable rows → {}", len(kept), args.out)

    logger.info("📊 summary:")
    for i, q, g, p, f1, b1, em in all_scores:
        marker = "✅" if args.lo < f1 < args.hi else " "
        logger.info("  {} #{:>2} F1={:.2f} EM={:.2f}  Q: {}", marker, i, f1, em, q[:60])


if __name__ == "__main__":
    main()
