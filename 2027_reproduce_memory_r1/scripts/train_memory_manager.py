"""RL fine-tune the Memory Manager. Config picks PPO vs. GRPO.

Usage::

    uv run python scripts/train_memory_manager.py configs/manager_grpo_h100_llama_8b.yaml

Logs → ``logs/<execution_id>/run.log``. ``logs/latest`` symlinks to the newest run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_r1.training.entrypoints import train_memory_manager
from memory_r1.utils import init_run_logger, logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()

    init_run_logger("train_memory_manager")
    logger.info("⚙️  config={}", args.config)
    try:
        train_memory_manager(args.config)
        logger.success("🏁 train_memory_manager done")
    except Exception:
        logger.exception("❌ train_memory_manager crashed")
        raise


if __name__ == "__main__":
    main()
