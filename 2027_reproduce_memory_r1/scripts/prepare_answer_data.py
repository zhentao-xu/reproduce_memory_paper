"""Algorithm 2: build Answer Agent training tuples.

Usage::

    uv run python scripts/prepare_answer_data.py \
        --locomo data/raw/locomo/locomo10.json \
        --out data/processed/answer_train.jsonl \
        --extractor openai --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_r1.agents.llm_backend import HFBackend, OpenAIBackend
from memory_r1.data.construction import build_answer_dataset
from memory_r1.utils import init_run_logger, logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default=Path("data/raw/locomo/locomo10.json"), type=Path)
    ap.add_argument("--out", default=Path("data/processed/answer_train.jsonl"), type=Path)
    ap.add_argument("--top-k-per-speaker", default=30, type=int)
    ap.add_argument("--extractor", default="heuristic", choices=["heuristic", "openai", "hf"])
    ap.add_argument("--extractor-model", default="gpt-4o-mini")
    ap.add_argument("--manager-backend", default="none", choices=["none", "openai", "hf"])
    ap.add_argument("--manager-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-dialogues", default=None, type=int)
    ap.add_argument(
        "--chunking",
        default="fact",
        choices=["fact", "turn_pair"],
        help="'fact' = per-turn extracted facts (paper). 'turn_pair' = one memory entry per "
             "overlapping consecutive-turn pair (sliding window, stride 1) so both A→B and "
             "B→A adjacencies are indexed — keeps Q+A together for GRPO/DPO.",
    )
    ap.add_argument(
        "--encoder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model. Try 'intfloat/e5-large-v2' for higher retrieval recall.",
    )
    args = ap.parse_args()

    init_run_logger("prepare_answer_data")
    logger.info(
        "⚙️  locomo={} out={} extractor={} encoder={} chunking={} top_k_per_speaker={} max_dialogues={}",
        args.locomo, args.out, args.extractor, args.encoder, args.chunking,
        args.top_k_per_speaker, args.max_dialogues,
    )

    if args.extractor == "openai":
        extractor_backend = OpenAIBackend(model=args.extractor_model)
    elif args.extractor == "hf":
        extractor_backend = HFBackend(model_name_or_path=args.extractor_model, dtype=args.dtype)
    else:
        extractor_backend = None

    if args.manager_backend == "openai":
        manager_backend = OpenAIBackend(model=args.manager_model)
    elif args.manager_backend == "hf":
        manager_backend = HFBackend(model_name_or_path=args.manager_model, dtype=args.dtype)
    else:
        manager_backend = None

    from memory_r1.memory.retrieval import DenseRetriever

    retriever = DenseRetriever(encoder_name=args.encoder)

    build_answer_dataset(
        locomo_path=args.locomo,
        out_path=args.out,
        top_k_per_speaker=args.top_k_per_speaker,
        manager_backend=manager_backend,
        extractor_backend=extractor_backend,
        retriever=retriever,
        max_dialogues=args.max_dialogues,
        chunking=args.chunking,
    )


if __name__ == "__main__":
    main()
