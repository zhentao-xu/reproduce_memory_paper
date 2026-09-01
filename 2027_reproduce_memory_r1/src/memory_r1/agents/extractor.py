"""FactExtractor — Algorithm 3 line 5: ``f_i <- LLMExtract(t_i)``.

Extracts atomic factual statements from each dialogue turn. In the paper, this is done by
GPT-4o-mini during data construction. At inference time (Algorithm 5, line 7) the same operation
is called, so we keep the class generic over any LLM backend.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from memory_r1.agents.llm_backend import ChatMessage, LLMBackend
from memory_r1.prompts.extractor import EXTRACTOR_SYSTEM, build_extractor_prompt


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ExtractedFacts:
    facts: list[str]
    raw: str


class FactExtractor:
    def __init__(self, backend: LLMBackend, temperature: float = 0.0, max_tokens: int = 512) -> None:
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(self, speaker: str, utterance: str, timestamp: str | None = None) -> ExtractedFacts:
        messages = [
            ChatMessage(role="system", content=EXTRACTOR_SYSTEM),
            ChatMessage(role="user", content=build_extractor_prompt(speaker, utterance, timestamp)),
        ]
        raw = self.backend.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        return ExtractedFacts(facts=self._parse(raw), raw=raw)

    @staticmethod
    def _parse(raw: str) -> list[str]:
        m = _JSON_RE.search(raw.strip())
        if not m:
            return []
        try:
            payload = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        facts = payload.get("facts", [])
        if not isinstance(facts, list):
            return []
        return [str(x).strip() for x in facts if str(x).strip()]
