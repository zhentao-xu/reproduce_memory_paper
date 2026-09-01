"""Algorithm 1: build Memory Manager training tuples.

Usage::

    # Fast, offline (heuristic fact extractor):
    uv run python scripts/prepare_manager_data.py \
        --locomo data/raw/locomo/locomo10.json \
        --out data/processed/manager_train.jsonl

    # Faithful to the paper: use GPT-4o-mini for extraction (needs OPENAI_API_KEY).
    uv run python scripts/prepare_manager_data.py \
        --locomo data/raw/locomo/locomo10.json \
        --out data/processed/manager_train.jsonl \
        --extractor openai --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_r1.agents.llm_backend import HFBackend, OpenAIBackend
from memory_r1.data.construction import build_manager_dataset, write_locomo_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default=Path("data/raw/locomo/locomo10.json"), type=Path)
    ap.add_argument("--out", default=Path("data/processed/manager_train.jsonl"), type=Path)
    ap.add_argument("--lookback-turns", default=24, type=int)
    ap.add_argument("--max-dialogues", default=None, type=int)
    ap.add_argument("--extractor", default="heuristic", choices=["heuristic", "openai", "hf"])
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--write-splits", action="store_true")
    args = ap.parse_args()

    if args.extractor == "openai":
        backend = OpenAIBackend(model=args.model)
    elif args.extractor == "hf":
        backend = HFBackend(model_name_or_path=args.model, dtype=args.dtype)
    else:
        backend = None

    build_manager_dataset(
        locomo_path=args.locomo,
        out_path=args.out,
        lookback_turns=args.lookback_turns,
        extractor_backend=backend,
        max_dialogues=args.max_dialogues,
    )

    if args.write_splits:
        write_locomo_splits(args.locomo, args.out.parent)


if __name__ == "__main__":
    main()
