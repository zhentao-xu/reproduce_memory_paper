"""Reward pipelines for the two agents.

Both rewards implement the paper's outcome-driven scheme (Section 3.1 / 3.2). The reward for the
*Memory Manager* is derived by:

1. Parsing the Manager's raw text into a ``ManagerOutput``.
2. Applying it to a *snapshot* of the temporal memory bank in the training example.
3. Running the frozen Answer Agent on the linked QA pair using the *updated* bank + retriever.
4. Computing EM against the gold answer.

Reward for the *Answer Agent* is much simpler: EM(y_pred, y_gold) directly.

We use plain Python callables — the trainer batches them via ``ThreadPoolExecutor`` when running
against a remote judge, or serially when using a local HF backend.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Mapping

from memory_r1.agents.answer_agent import AnswerAgent
from memory_r1.eval.metrics import em_reward, token_f1
from memory_r1.memory.bank import MemoryBank
from memory_r1.memory.operations import (
    ManagerOutput,
    apply_manager_output,
    parse_manager_output,
)
from memory_r1.memory.retrieval import DenseRetriever


ManagerRewardFn = Callable[[str, "ManagerTrainSample"], float]
AnswerAgentRewardFn = Callable[[str, "AnswerTrainSample"], float]


# --------------------------------------------------------------------------- samples


@dataclass
class ManagerTrainSample:
    """One (state, gold) tuple driving the Manager's reward.

    - ``speaker``: the speaker of the current turn.
    - ``facts``: newly extracted facts (the Manager's input).
    - ``old_bank_state``: bank state *before* the Manager acts. Serialized dict.
    - ``qa``: the linked QA pair (question + gold). If multiple QAs are linked to this turn, we use
      the first one — training over all of them multiplies the same reward signal.
    - ``turn_timestamp``: passed to ADD-inserted entries.
    """

    speaker: str
    facts: list[str]
    old_bank_state: dict
    qa: dict  # {"question", "answer"}
    turn_timestamp: str | None = None


@dataclass
class AnswerTrainSample:
    question: str
    gold_answer: str
    retrieved: Mapping[str, list[dict]]  # {speaker -> [{"id","text","timestamp"}]}


# --------------------------------------------------------------------------- reward for Manager


def build_manager_reward_fn(
    answer_agent: AnswerAgent,
    retriever: DenseRetriever,
    top_k_per_speaker: int = 30,
    on_parse_error: float = 0.0,
) -> Callable[[str, ManagerTrainSample], float]:
    """Return a ``fn(raw_manager_text, sample) -> float`` reward.

    Steps mirror Section 3.1 Reward Design:

    1. Restore ``old_bank`` from ``sample.old_bank_state``.
    2. Parse the Manager's text. If malformed, return ``on_parse_error`` (default 0).
    3. Apply the ops to ``old_bank`` for ``sample.speaker``.
    4. Retrieve 60 memories for ``sample.qa['question']``.
    5. Ask the frozen Answer Agent for a prediction.
    6. Return ``EM(pred, gold)``.
    """

    def _reward(raw_manager: str, sample: ManagerTrainSample) -> float:
        try:
            output: ManagerOutput = parse_manager_output(raw_manager)
        except Exception:
            return on_parse_error

        bank = MemoryBank.from_dict(sample.old_bank_state)
        # Manager sees a top-K old-memory view; we approximate by using the full per-speaker view
        # here since ``sample.old_bank_state`` is already the truncated view built at data-prep
        # time.
        old_view = bank.as_prompt_list(sample.speaker)
        apply_manager_output(
            speaker=sample.speaker,
            old_memory_view=old_view,
            output=output,
            bank=bank,
            turn_timestamp=sample.turn_timestamp,
        )

        # Answer the linked QA on the updated bank.
        question = sample.qa["question"]
        gold = sample.qa["answer"]
        retrieved = retriever.search_by_speaker(bank, question, top_k_per_speaker=top_k_per_speaker)
        mem_by_speaker = {
            sp: [{"id": h.entry_id, "text": h.text, "timestamp": h.timestamp or ""} for h in hits]
            for sp, hits in retrieved.items()
        }
        out = answer_agent.answer(question=question, memories_by_speaker=mem_by_speaker)
        return em_reward(out.answer, gold)

    return _reward


# --------------------------------------------------------------------------- reward for Answer


def build_answer_reward_fn(
    reward_type: str = "em",
) -> Callable[[str, AnswerTrainSample], float]:
    """Return the Answer-Agent reward function.

    ``em`` (default, paper's Table 2 winner), ``f1``, or ``judge`` (uses ``LLMJudge`` — expensive).
    """

    if reward_type == "em":
        def _r(raw_answer: str, sample: AnswerTrainSample) -> float:
            from memory_r1.agents.answer_agent import AnswerAgent

            out = AnswerAgent._parse(raw_answer)
            return em_reward(out.answer, sample.gold_answer)

        return _r

    if reward_type == "f1":
        def _r(raw_answer: str, sample: AnswerTrainSample) -> float:
            from memory_r1.agents.answer_agent import AnswerAgent

            out = AnswerAgent._parse(raw_answer)
            return token_f1(out.answer, sample.gold_answer)

        return _r

    if reward_type == "judge":
        from memory_r1.eval.judge import LLMJudge

        judge = LLMJudge()

        def _r(raw_answer: str, sample: AnswerTrainSample) -> float:
            from memory_r1.agents.answer_agent import AnswerAgent

            out = AnswerAgent._parse(raw_answer)
            return float(judge.judge(sample.question, sample.gold_answer, out.answer).correct)

        return _r

    raise ValueError(f"Unknown reward_type: {reward_type}")


# --------------------------------------------------------------------------- batching helpers


def score_manager_batch(
    reward_fn: Callable[[str, ManagerTrainSample], float],
    raws: list[str],
    samples: list[ManagerTrainSample],
    max_workers: int = 4,
) -> list[float]:
    """Compute rewards in parallel threads (useful when the Answer Agent is a remote API)."""

    assert len(raws) == len(samples)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(reward_fn, raws, samples))


def score_answer_batch(
    reward_fn: Callable[[str, AnswerTrainSample], float],
    raws: list[str],
    samples: list[AnswerTrainSample],
    max_workers: int = 4,
) -> list[float]:
    assert len(raws) == len(samples)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(reward_fn, raws, samples))
