"""LLM-as-a-Judge (Figure 12 prompt) — J metric in the paper.

We use GPT-4o-mini by default. Any OpenAI-compatible endpoint works.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from memory_r1.agents.llm_backend import ChatMessage, LLMBackend, OpenAIBackend
from memory_r1.prompts.judge import build_judge_prompt


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeResult:
    label: str  # "CORRECT" | "WRONG"
    correct: bool
    explanation: str
    raw: str


class LLMJudge:
    def __init__(self, backend: LLMBackend | None = None, model: str = "gpt-4o-mini") -> None:
        self.backend = backend or OpenAIBackend(model=model)

    def judge(self, question: str, gold_answer: str, generated_answer: str) -> JudgeResult:
        prompt = build_judge_prompt(question, gold_answer, generated_answer)
        raw = self.backend.chat(
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=256,
        )
        label = self._extract_label(raw)
        correct = label.upper() == "CORRECT"
        explanation = self._first_sentence(raw)
        return JudgeResult(label=label, correct=correct, explanation=explanation, raw=raw)

    @staticmethod
    def _extract_label(raw: str) -> str:
        # Prefer the JSON block per the prompt's instruction.
        m = _JSON_RE.search(raw)
        if m:
            try:
                payload = json.loads(m.group(0))
                if "label" in payload:
                    return str(payload["label"]).strip().upper()
            except json.JSONDecodeError:
                pass
        # Fall back: look for CORRECT / WRONG token.
        upper = raw.upper()
        if "CORRECT" in upper and "WRONG" not in upper:
            return "CORRECT"
        if "WRONG" in upper and "CORRECT" not in upper:
            return "WRONG"
        # Ambiguous — the paper's prompt warns not to include both.
        return "WRONG"

    @staticmethod
    def _first_sentence(raw: str) -> str:
        first = raw.strip().split("\n", 1)[0]
        return first[:512]


def judge_batch(
    judge: LLMJudge,
    triples: list[tuple[str, str, str]],
) -> list[JudgeResult]:
    """Serial judge over ``[(question, gold, pred)]``. Wrap in threads if you need parallelism."""

    return [judge.judge(q, g, p) for q, g, p in triples]
