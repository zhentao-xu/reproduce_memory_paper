"""Answer Agent output parser tests."""

from __future__ import annotations

from memory_r1.agents.answer_agent import AnswerAgent


def test_parse_with_marker():
    raw = (
        "**Memories selected as relevant:**\n"
        "- 8:30 pm on 1 January, 2023: John has a nostalgic memory of the beach.\n"
        "- 1:24 pm on 25 May, 2023: John shared a picture of his family at the beach.\n\n"
        "**Answer:** beach"
    )
    out = AnswerAgent._parse(raw)
    assert out.answer == "beach"
    assert len(out.distilled_memories) == 3  # includes the header line


def test_parse_no_marker_falls_back():
    raw = "The answer is: mountains."
    out = AnswerAgent._parse(raw)
    assert out.answer.lower().startswith("mountains")


def test_parse_strips_quotes_and_prefix():
    raw = "**Answer:** \"beach\""
    assert AnswerAgent._parse(raw).answer == "beach"
