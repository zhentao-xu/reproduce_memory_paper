"""Download LoCoMo, MSC, and LongMemEval into ``data/raw/``.

Usage::

    uv run python scripts/download_datasets.py --dataset all
    uv run python scripts/download_datasets.py --dataset locomo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_r1.data.download import download_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", "locomo", "msc", "longmemeval"])
    ap.add_argument("--out-dir", default="data/raw", type=Path)
    args = ap.parse_args()
    download_all(dataset=args.dataset, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
