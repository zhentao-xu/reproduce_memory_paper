"""Verify that the local e5-small-v2 checkpoint loads and encodes correctly.

Usage:
    python scripts/verify_e5.py
    python scripts/verify_e5.py --model-dir /custom/path/to/e5-small-v2

Passes if:
    - SentenceTransformer loads the directory
    - Embedding dimension is 384 (e5-small-v2 spec)
    - Cosine similarity ranks the semantically-matching passage above distractors,
      using the "query: " / "passage: " prefixes the model was trained with.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "e5-small-v2"
)
EXPECTED_DIM = 384

QUERY = "How do I train a memory manager with GRPO?"
PASSAGES = [
    "The Memory Manager is trained with GRPO on retrieval-augmented rollouts.",
    "Golden retrievers are friendly dogs that love to swim.",
    "A tomato is technically a fruit but used as a vegetable in cooking.",
]
EXPECTED_TOP = 0  # index of the correct passage in PASSAGES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Local model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument("--device", default=None, help="cpu / cuda / mps (default: auto)")
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
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"❌ sentence-transformers not installed: {e}")
        return 1

    print("→ loading SentenceTransformer …")
    try:
        model = SentenceTransformer(str(model_dir), device=args.device)
    except Exception as e:
        print(f"❌ load failed: {type(e).__name__}: {e}")
        return 1

    dim = int(model.get_sentence_embedding_dimension())
    print(f"✓ loaded — device={model.device}, embedding dim={dim}")
    if dim != EXPECTED_DIM:
        print(f"❌ expected dim {EXPECTED_DIM} for e5-small-v2, got {dim}")
        return 1

    print("→ encoding query + passages with e5 prefixes …")
    q_emb = model.encode(
        [f"query: {QUERY}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    p_emb = model.encode(
        [f"passage: {p}" for p in PASSAGES],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    print(f"✓ query shape={q_emb.shape}, passages shape={p_emb.shape}")

    sims = (q_emb @ p_emb.T)[0]
    ranking = np.argsort(-sims)
    print("→ cosine similarities (higher = more similar):")
    for i, p in enumerate(PASSAGES):
        marker = "★" if i == EXPECTED_TOP else " "
        print(f"   {marker} [{sims[i]:+.4f}] {p}")

    if int(ranking[0]) != EXPECTED_TOP:
        print(f"❌ semantic sanity check failed — expected passage #{EXPECTED_TOP} on top, "
              f"got #{ranking[0]}")
        return 1

    print("\n✅ verification passed — model loads, dim matches, retrieval ranking is sane.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
