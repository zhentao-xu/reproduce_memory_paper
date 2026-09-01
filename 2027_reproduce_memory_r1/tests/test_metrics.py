"""Test EM / F1 / BLEU-1 metric implementations."""

from __future__ import annotations

from memory_r1.eval.metrics import bleu1, exact_match, normalize_answer, token_f1


def test_normalize_strips_articles_and_punct():
    assert normalize_answer("The Beach.") == "beach"
    assert normalize_answer("A shell necklace!") == "shell necklace"


def test_exact_match():
    assert exact_match("beach", "The beach.") == 1.0
    assert exact_match("mountains", "beach") == 0.0


def test_token_f1_perfect_and_partial():
    assert token_f1("shell necklace", "a shell necklace") == 1.0
    # Partial overlap: 1/2 precision, 1/1 recall -> F1 = 2*0.5*1/(0.5+1) = 0.6667
    assert abs(token_f1("shell keychain", "shell") - 0.6667) < 0.01


def test_bleu1_runs():
    # We just check the function doesn't blow up and returns a sensible range.
    v = bleu1("shell necklace", "a shell necklace")
    assert 0.0 <= v <= 1.0
