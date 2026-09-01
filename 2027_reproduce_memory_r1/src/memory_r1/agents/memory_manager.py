"""MemoryManager — chooses ADD/UPDATE/DELETE/NOOP and updates the memory bank.

Implements Algorithm 3 (Memory Bank Construction) at inference time and Algorithm 5 (Memory Manager
training loop) inside the RL trainer.
"""

from __future__ import annotations

from dataclasses import dataclass

from memory_r1.agents.llm_backend import ChatMessage, LLMBackend
from memory_r1.memory.bank import MemoryBank
from memory_r1.memory.operations import ManagerOutput, apply_manager_output, parse_manager_output
from memory_r1.memory.retrieval import DenseRetriever
from memory_r1.prompts.manager import MEMORY_MANAGER_SYSTEM, build_manager_prompt


@dataclass
class ManagerStep:
    """Diagnostic record for a single Manager rollout step."""

    speaker: str
    facts: list[str]
    old_memory_view: list[dict[str, str]]
    prompt_user: str
    raw_output: str
    parsed: ManagerOutput | None
    counts: dict[str, int]


class MemoryManager:
    """Stateless wrapper: given a bank + facts, mutate the bank via the policy LLM."""

    def __init__(
        self,
        backend: LLMBackend,
        retriever: DenseRetriever | None = None,
        top_k_context: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.backend = backend
        self.retriever = retriever
        self.top_k_context = top_k_context
        self.temperature = temperature
        self.max_tokens = max_tokens

    def step(
        self,
        bank: MemoryBank,
        speaker: str,
        facts: list[str],
        turn_timestamp: str | None = None,
    ) -> ManagerStep:
        """Ask the policy to update the memory for ``speaker`` given the new ``facts``.

        Following Algorithm 3 line 5-6, we first retrieve the top-K most similar existing entries
        for ``speaker`` as the ``old_memory`` view — this keeps the prompt small even when the
        bank has grown to hundreds of entries. If no retriever is provided, we pass the full
        per-speaker view.
        """

        query = " ; ".join(facts) if facts else ""
        if self.retriever is not None and query:
            hits = self.retriever.search_bank(bank, query=query, top_k=self.top_k_context, speaker=speaker)
            old_view = [{"id": h.entry_id, "text": h.text} for h in hits]
        else:
            old_view = bank.as_prompt_list(speaker)

        prompt_user = build_manager_prompt(retrieved_facts=facts, old_memory=old_view)

        raw = self.backend.chat(
            messages=[
                ChatMessage(role="system", content=MEMORY_MANAGER_SYSTEM),
                ChatMessage(role="user", content=prompt_user),
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        parsed: ManagerOutput | None
        counts: dict[str, int]
        try:
            parsed = parse_manager_output(raw)
            counts = apply_manager_output(
                speaker=speaker,
                old_memory_view=old_view,
                output=parsed,
                bank=bank,
                turn_timestamp=turn_timestamp,
            )
        except Exception:
            parsed = None
            counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}

        return ManagerStep(
            speaker=speaker,
            facts=facts,
            old_memory_view=old_view,
            prompt_user=prompt_user,
            raw_output=raw,
            parsed=parsed,
            counts=counts,
        )
