"""Answer Agent prompt.

The paper (Figure 11) uses a 10-rule system prompt plus an 8-step "APPROACH" section. On modern
instruct models like Qwen3-4B, that dense-rule format tends to be **ignored** (the model reads the
retrieved memory timestamp literally and skips the "convert relative references" instruction).

We ship two variants:

- ``ANSWER_AGENT_SYSTEM_PAPER`` — verbatim paper Figure 11 (kept for exact-reproduction runs).
- ``ANSWER_AGENT_SYSTEM_FOCUSED`` — shorter, action-oriented prompt that empirically works better on
  Qwen3-family models. Default at ``ANSWER_AGENT_SYSTEM``.

Switch between them via ``AnswerAgent(system_prompt=ANSWER_AGENT_SYSTEM_PAPER)`` or by patching the
module attribute in a config.
"""

from __future__ import annotations

from typing import Iterable, Mapping


# --------------------------------------------------------------------------- paper Figure 11

ANSWER_AGENT_SYSTEM_PAPER = """\
You are an intelligent memory assistant tasked with retrieving
accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation.
These memories contain timestamped information that may be relevant
to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. May special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago"),
    calculate the actual date based on the memory timestamp.
6. Always convert relative time references to specific dates, months, or years.
7. Focus only on the content of the memories. Do not confuse character names
8. The answer should be less than 5-6 words.
9. IMPORTANT: Select memories you found that are useful for answering the questions,
and output it before you answer questions.
10. IMPORTANT: Output the final answer after **Answer:**

# APPROACH (Think step by step):
1. Examine all relevant memories
2. Examine the timestamps carefully
3. Look for explicit mentions that answer the question
4. Convert relative references if needed
5. Formulate a concise answer
6. Double-check the answer correctness
7. Ensure the final answer is specific
8. First output the memories that you found are important before you answer questions
"""


# --------------------------------------------------------------------------- focused (default)

ANSWER_AGENT_SYSTEM_FOCUSED = """\
You answer questions about a conversation by reading the memories provided.

Rules:
- Each memory has a timestamp (e.g. "1:56 pm on 8 May, 2023"). Use it.
- If a memory says "yesterday", "last year", "two months ago", convert it to an actual date
  relative to the memory's timestamp. Do NOT copy the memory timestamp as the answer if the
  memory uses a relative reference.
- Only use facts stated in the memories. Do not invent.
- Keep the answer under 6 words.

Output format:
  **Answer:** <your final answer, under 6 words>
"""


# Default — the focused prompt. Override via ``AnswerAgent(system_prompt=ANSWER_AGENT_SYSTEM_PAPER)``
# to reproduce the paper's exact behavior.
ANSWER_AGENT_SYSTEM = ANSWER_AGENT_SYSTEM_FOCUSED


# --------------------------------------------------------------------------- helpers


def _format_speaker_memories(name: str, memories: Iterable[Mapping[str, str]]) -> str:
    """Format one speaker's memories as bulleted timestamped lines.

    Each memory is expected to have at least ``text``; ``timestamp`` and ``speaker`` are optional.
    """

    lines = [f"Memories for user {name}:"]
    for mem in memories:
        text = str(mem.get("text", "")).strip()
        if not text:
            continue
        ts = str(mem.get("timestamp", "")).strip()
        prefix = f"- {ts}: " if ts else "- "
        lines.append(f"{prefix}{text}")
    return "\n".join(lines)


def build_answer_prompt(
    question: str,
    memories_by_speaker: Mapping[str, list[Mapping[str, str]]],
) -> str:
    """Assemble the user-turn prompt with per-speaker memory lists and the question."""

    sections = [
        _format_speaker_memories(name, mems) for name, mems in memories_by_speaker.items()
    ]
    joined = "\n\n".join(sections)
    return f"{joined}\n\nQuestion: {question}"
