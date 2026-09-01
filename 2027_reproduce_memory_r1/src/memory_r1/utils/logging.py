"""Loguru-based logging with per-execution log folders.

Each script call creates ``logs/<execution_id>/`` containing:

- ``run.log`` — colored, all levels (readable via ``less -R``)
- ``run.jsonl`` — structured records for programmatic access

``logs/latest`` symlinks to the newest run. In another terminal::

    tail -F logs/latest/run.log

Icons: hard-code them in the log string. Palette used in this codebase::

    🚀 start   🏁 done   ⚙️  config   📥 load    🧠 model   🖥️  device
    📚 data    🏋️  train  🎲 rollout   🎯 reward  📉 grad    🔄 step
    📊 eval    📈 metric  💾 save     ✅ ok      ⚠️  warn    ❌ error
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def _short_hash(n: int = 6) -> str:
    return hashlib.sha1(f"{os.getpid()}-{datetime.now().isoformat()}".encode()).hexdigest()[:n]


def init_run_logger(
    script_name: str,
    log_root: Path | str = "logs",
    level: str = "INFO",
) -> tuple[str, Path]:
    """Configure loguru sinks + return ``(execution_id, log_dir)``."""

    root = Path(log_root)
    root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    execution_id = f"{ts}-{script_name}-{_short_hash()}"
    log_dir = root / execution_id
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    fmt = (
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}:{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, colorize=True, format=fmt)
    # Colored file — ``less -R`` renders it. Keep colors for grepability of levels too.
    logger.add(str(log_dir / "run.log"), level="DEBUG", colorize=True, format=fmt,
               rotation="100 MB", retention=5, enqueue=True)
    # Structured JSONL — one dict per record, machine-readable.
    logger.add(str(log_dir / "run.jsonl"), level="DEBUG", serialize=True,
               rotation="100 MB", retention=5, enqueue=True)

    # Level colors — loguru's ``<level>`` tag looks these up.
    for name, color in (
        ("DEBUG",    "<blue>"),
        ("INFO",     "<white>"),
        ("SUCCESS",  "<green><bold>"),
        ("WARNING",  "<yellow><bold>"),
        ("ERROR",    "<red><bold>"),
        ("CRITICAL", "<magenta><bold>"),
    ):
        try:
            logger.level(name, color=color)
        except ValueError:
            pass

    # ``logs/latest`` symlink.
    latest = root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(execution_id)
    except OSError:
        pass

    logger.info("🚀 execution_id={} log_dir={}", execution_id, log_dir)
    logger.info("ℹ️  tail -F {}/run.log", log_dir)
    return execution_id, log_dir


__all__ = ["init_run_logger", "logger"]
