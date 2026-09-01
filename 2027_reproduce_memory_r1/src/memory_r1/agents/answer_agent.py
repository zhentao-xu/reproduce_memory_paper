"""AnswerAgent — takes 60 retrieved memories + question, does Memory Distillation, then answers.

The paper's Answer-Agent output has two parts (Figure 11 instruction #9 + #10):

1. **Memories selected as relevant**: bulleted subset of the 60 candidates.
2. **Answer**: <= 5-6 words, after the literal string ``**Answer:**``.

We parse both back out for downstream metrics and for the exact-match reward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

from memory_r1.agents.llm_backend import ChatMessage, LLMBackend
from memory_r1.prompts.answer import ANSWER_AGENT_SYSTEM, build_answer_prompt


ANSWER_MARKER = "**Answer:**"


@dataclass
class AnswerOutput:
    answer: str
    distilled_memories: list[str] = field(default_factory=list)
    raw: str = ""
    prompt_user: str = ""


class AnswerAgent:
    """Wraps the answer LLM. The class is stateless; state lives in the bank + retriever."""

    def __init__(
        self,
        backend: LLMBackend,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> None:
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Default to the module-level ``ANSWER_AGENT_SYSTEM`` (currently the focused variant).
        # Callers can pass ``ANSWER_AGENT_SYSTEM_PAPER`` to reproduce the paper exactly.
        self.system_prompt = system_prompt or ANSWER_AGENT_SYSTEM

    def answer(
        self,
        question: str,
        memories_by_speaker: Mapping[str, list[dict[str, str]]],
    ) -> AnswerOutput:
        prompt_user = build_answer_prompt(question=question, memories_by_speaker=memories_by_speaker)
        raw = self.backend.chat(
            messages=[
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=prompt_user),
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self._parse(raw, prompt_user=prompt_user)

    @staticmethod
    def _parse(raw: str, prompt_user: str = "") -> AnswerOutput:
        """Split model output into (distilled memories, answer).

        The prompt instructs the model to write the distilled memories block first, then
        ``**Answer:**`` followed by the concise answer. We accept any prose above the marker as
        distilled memories, and normalize the answer by stripping surrounding whitespace and
        trailing periods.
        """

        text = raw.strip()
        answer = ""
        distilled_block = ""

        # Accept the paper's exact marker (``**Answer:**``) or the more common plain ``Answer:``
        # that Qwen / LLaMA emit without the bold. Split on the last such marker so any
        # intermediate "Answer:" inside a distilled memory doesn't confuse us.
        marker_match = re.search(r"(?:\*\*Answer:?\*\*|(?<![A-Za-z])Answer\s*:)", text, flags=re.IGNORECASE)
        if marker_match:
            # Look for the LAST occurrence, safer against distractor "Answer:" strings.
            matches = list(re.finditer(r"(?:\*\*Answer:?\*\*|(?<![A-Za-z])Answer\s*:)", text, flags=re.IGNORECASE))
            m = matches[-1]
            distilled_block = text[: m.start()]
            answer = text[m.end():].strip()
        else:
            answer = text

        # Trim trailing full-stops/quotes for cleaner EM.
        answer = answer.strip().strip('"').strip("'").strip()
        # If the answer starts with prose like "The answer is X.", strip the prefix.
        answer = re.sub(r"^(?:the )?answer(?:\s*is)?[:\-]?\s*", "", answer, flags=re.IGNORECASE)

        distilled = [
            line.strip(" -*•\t") for line in distilled_block.splitlines() if line.strip(" -*•\t")
        ]

        return AnswerOutput(
            answer=answer,
            distilled_memories=distilled,
            raw=raw,
            prompt_user=prompt_user,
        )
