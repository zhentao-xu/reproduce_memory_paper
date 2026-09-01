"""RL fine-tune the Answer Agent.

Usage::

    uv run python scripts/train_answer_agent.py configs/answer_grpo_h100_llama_8b.yaml

Logs → ``logs/<execution_id>/run.log`` (colored, ``less -R``-friendly). ``logs/latest`` symlinks
to the newest run. To watch live::

    tail -F logs/latest/run.log
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_r1.training.entrypoints import train_answer_agent
from memory_r1.utils import init_run_logger, logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()

    init_run_logger("train_answer_agent")
    logger.info("⚙️  config={}", args.config)
    try:
        train_answer_agent(args.config)
        logger.success("🏁 train_answer_agent done")
    except Exception:
        logger.exception("❌ train_answer_agent crashed")
        raise


if __name__ == "__main__":
    main()
