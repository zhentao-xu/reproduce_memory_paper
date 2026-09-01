"""Prompt formatting round-trips."""

from __future__ import annotations

from memory_r1.prompts import (
    ANSWER_AGENT_SYSTEM,
    EXTRACTOR_SYSTEM,
    JUDGE_SYSTEM,
    MEMORY_MANAGER_SYSTEM,
    build_answer_prompt,
    build_judge_prompt,
    build_manager_prompt,
)


def test_manager_prompt_contains_facts_and_ids():
    p = build_manager_prompt(
        retrieved_facts=["Alice loves pizza"],
        old_memory=[{"id": "0", "text": "Alice is a software engineer"}],
    )
    assert "Alice loves pizza" in p
    assert "software engineer" in p
    assert '"id"' in p


def test_manager_system_prompt_mentions_all_ops():
    for op in ("ADD", "UPDATE", "DELETE", "NONE"):
        assert op in MEMORY_MANAGER_SYSTEM


def test_answer_prompt_lists_memories_per_speaker():
    p = build_answer_prompt(
        question="Where does John live?",
        memories_by_speaker={
            "John": [{"text": "John lives near the beach.", "timestamp": "2023-05-25"}],
            "Maria": [{"text": "Maria takes photos.", "timestamp": ""}],
        },
    )
    assert "Memories for user John" in p
    assert "beach" in p
    assert "Question: Where does John live?" in p
    # Focused prompt uses "under 6 words"; paper's uses "5-6 words". Both encode the
    # length constraint from paper Figure 11.
    assert "6 words" in ANSWER_AGENT_SYSTEM


def test_judge_prompt_formats_all_fields():
    p = build_judge_prompt("Q?", "A", "B")
    assert "Q?" in p and "A" in p and "B" in p and "CORRECT" in JUDGE_SYSTEM


def test_extractor_system_asks_for_json():
    assert "JSON" in EXTRACTOR_SYSTEM
    assert "facts" in EXTRACTOR_SYSTEM
