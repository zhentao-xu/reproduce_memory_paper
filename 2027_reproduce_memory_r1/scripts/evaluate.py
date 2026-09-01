"""Run the full pipeline on a benchmark and print F1/BLEU-1/EM/J.

Usage::

    uv run python scripts/evaluate.py configs/eval_h100_llama_8b.yaml

Logs → ``logs/<execution_id>/run.log``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_r1.eval.evaluator import run_evaluation
from memory_r1.utils import init_run_logger, logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()

    init_run_logger("evaluate")
    logger.info("⚙️  config={}", args.config)
    try:
        run_evaluation(args.config)
        logger.success("🏁 evaluation done")
    except Exception:
        logger.exception("❌ evaluation crashed")
        raise


if __name__ == "__main__":
    main()
