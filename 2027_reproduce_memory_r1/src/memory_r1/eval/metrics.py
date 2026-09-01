"""EM / F1 / BLEU-1 metrics used for training rewards and eval reporting.

The paper reports three metrics for LoCoMo-style QA:

- **F1**: token-level F1 between the generated and gold answer.
- **B1**: BLEU-1 (unigram) via ``sacrebleu``.
- **J**: LLM-as-a-Judge (see :mod:`memory_r1.eval.judge`).

The RL reward is **EM** (exact match after normalization) per the paper's Section 3.1 Reward
Design. This module provides all three plus a common normalizer.
"""

from __future__ import annotations

import re
import string
from collections import Counter


# --------------------------------------------------------------------------- normalization


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WHITESPACE = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace.

    Mirrors the SQuAD/HuggingFace ``squad_v2`` normalizer, which is what Mem0 (the baseline
    Memory-R1 compares against) and the LoCoMo eval scripts use.
    """

    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


# --------------------------------------------------------------------------- metrics


def exact_match(pred: str, gold: str) -> float:
    """Return 1.0 if normalized pred == normalized gold, else 0.0."""

    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between the two normalized strings."""

    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1(pred: str, gold: str) -> float:
    """Corpus-level BLEU-1 via ``sacrebleu`` treating each pair as a single sentence.

    We use ``sacrebleu.sentence_bleu`` with ``max_order=1`` (unigram) and no smoothing (following
    Mem0's evaluation code).
    """

    try:
        from sacrebleu.metrics import BLEU

        bleu = BLEU(max_ngram_order=1, effective_order=True)
        score = bleu.sentence_score(hypothesis=pred, references=[gold])
        return float(score.score) / 100.0
    except Exception:
        # Fallback: manual unigram precision (BLEU-1 without brevity penalty).
        pred_tokens = pred.split()
        gold_tokens = gold.split()
        if not pred_tokens:
            return 0.0
        matches = sum(1 for t in pred_tokens if t in gold_tokens)
        return matches / len(pred_tokens)


# --------------------------------------------------------------------------- reward helpers


def em_reward(pred: str, gold: str) -> float:
    """Training reward for both Memory Manager (via Answer Agent) and Answer Agent — EM in [0, 1]."""

    return exact_match(pred, gold)


def scores_batch(preds: list[str], golds: list[str]) -> dict[str, float]:
    """Return {'em', 'f1', 'bleu1'} averaged over the batch (0..1 scale)."""

    assert len(preds) == len(golds)
    if not preds:
        return {"em": 0.0, "f1": 0.0, "bleu1": 0.0}
    em = sum(exact_match(p, g) for p, g in zip(preds, golds)) / len(preds)
    f1 = sum(token_f1(p, g) for p, g in zip(preds, golds)) / len(preds)
    b1 = sum(bleu1(p, g) for p, g in zip(preds, golds)) / len(preds)
    return {"em": em, "f1": f1, "bleu1": b1}
