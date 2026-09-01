"""LoCoMo dataset loader.

The ``locomo10.json`` release consists of 10 long multi-session dialogues (each with ~300 turns
across ~35 sessions). Each dialogue has:

- ``sample_id``: dialogue identifier.
- ``conversation``: session-keyed dict (``session_1``, ``session_2``, ...). Each session is a list
  of ``{"speaker": ..., "text": ..., "dia_id": ...}`` turns, sometimes with ``blip_caption`` or
  ``img_url`` (multimodal; we drop those for now).
- ``qa``: list of QA items with ``question``, ``answer``, ``category`` (1=single-hop, 2=multi-hop,
  3=temporal, 4=open-domain, 5=adversarial). Following the paper we exclude category 5.

We normalize this into a stream of ``LoCoMoTurn`` objects (in chronological order) plus a list of
``LoCoMoQA`` per dialogue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CATEGORY_NAMES = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}


@dataclass
class LoCoMoTurn:
    """One utterance from a session, in chronological order across the whole dialogue."""

    dialogue_id: str
    session_id: str
    session_index: int
    turn_index_in_session: int
    turn_index_global: int
    speaker: str
    text: str
    timestamp: str | None = None  # session-level datetime, if available
    dia_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoCoMoQA:
    dialogue_id: str
    question: str
    answer: str
    category: int
    category_name: str
    evidence: list[str] = field(default_factory=list)
    adversarial: bool = False


@dataclass
class LoCoMoDialogue:
    dialogue_id: str
    turns: list[LoCoMoTurn]
    qa: list[LoCoMoQA]
    speakers: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class LoCoMoLoader:
    """Parse ``locomo10.json`` into structured objects.

    Also implements the paper's 1:1:8 train/val/test QA split (152/81/1307) with a fixed seed for
    reproducibility.
    """

    def __init__(self, path: str | Path, exclude_adversarial: bool = True) -> None:
        self.path = Path(path)
        self.exclude_adversarial = exclude_adversarial
        with self.path.open(encoding="utf-8") as f:
            self._raw = json.load(f)
        if not isinstance(self._raw, list):
            raise ValueError(f"Expected a list of dialogues in {self.path}, got {type(self._raw)}")
        self.dialogues: list[LoCoMoDialogue] = [self._parse_dialogue(d) for d in self._raw]

    # ------------------------------------------------------------------ parsing

    def _parse_dialogue(self, d: dict[str, Any]) -> LoCoMoDialogue:
        did = str(d.get("sample_id") or d.get("dialogue_id") or d.get("id"))
        convo = d.get("conversation", {})
        speakers_in_convo = set()

        # Sessions are keyed ``session_1``, ``session_2``, ..., alongside sibling metadata keys
        # like ``session_1_date_time``. Filter to *strict* ``session_<int>`` matches only.
        def _session_index(k: str) -> int | None:
            parts = k.split("_")
            if len(parts) != 2 or parts[0] != "session":
                return None
            try:
                return int(parts[1])
            except ValueError:
                return None

        session_keys = sorted(
            (k for k in convo.keys() if _session_index(k) is not None),
            key=lambda k: _session_index(k) or 0,
        )

        turns: list[LoCoMoTurn] = []
        global_idx = 0
        for si, sk in enumerate(session_keys):
            session_data = convo[sk]
            ts_key = f"{sk}_date_time"
            timestamp = convo.get(ts_key)  # LoCoMo stores per-session date-time siblings
            if not isinstance(session_data, list):
                # Sometimes wrapped in {"turns": [...]}.
                session_data = session_data.get("turns", []) if isinstance(session_data, dict) else []
            for ti, turn in enumerate(session_data):
                speaker = str(turn.get("speaker", "")).strip()
                text = str(turn.get("text") or turn.get("clean_text") or "").strip()
                if not text:
                    continue
                speakers_in_convo.add(speaker)
                turns.append(
                    LoCoMoTurn(
                        dialogue_id=did,
                        session_id=sk,
                        session_index=si + 1,
                        turn_index_in_session=ti,
                        turn_index_global=global_idx,
                        speaker=speaker,
                        text=text,
                        timestamp=timestamp,
                        dia_id=turn.get("dia_id"),
                        metadata={
                            k: v
                            for k, v in turn.items()
                            if k not in {"speaker", "text", "clean_text", "dia_id"}
                        },
                    )
                )
                global_idx += 1

        qa_items: list[LoCoMoQA] = []
        for q in d.get("qa", []):
            category = int(q.get("category", 0))
            adversarial = category == 5
            if adversarial and self.exclude_adversarial:
                continue
            answer = q.get("answer")
            if answer is None:
                # LoCoMo also has "adversarial_answer" for cat=5 items.
                answer = q.get("adversarial_answer", "")
            qa_items.append(
                LoCoMoQA(
                    dialogue_id=did,
                    question=str(q.get("question", "")),
                    answer=str(answer),
                    category=category,
                    category_name=CATEGORY_NAMES.get(category, str(category)),
                    evidence=list(q.get("evidence", []) or []),
                    adversarial=adversarial,
                )
            )

        return LoCoMoDialogue(
            dialogue_id=did,
            turns=turns,
            qa=qa_items,
            speakers=sorted(speakers_in_convo),
            metadata={k: v for k, v in d.items() if k not in {"conversation", "qa"}},
        )

    # ------------------------------------------------------------------ helpers

    def all_qa(self) -> list[tuple[LoCoMoDialogue, LoCoMoQA]]:
        return [(d, q) for d in self.dialogues for q in d.qa]

    def qa_split_152_81_1307(
        self, seed: int = 42
    ) -> tuple[
        list[tuple[LoCoMoDialogue, LoCoMoQA]],
        list[tuple[LoCoMoDialogue, LoCoMoQA]],
        list[tuple[LoCoMoDialogue, LoCoMoQA]],
    ]:
        """Paper's exact 152/81/1307 QA split.

        LoCoMo's ``locomo10.json`` file order yields the paper's split by dialogue:
        ``conv-26`` (152 QA) → train, ``conv-30`` (81 QA) → val, ``conv-41..conv-50`` (1307 QA
        total) → test. The counts fall out exactly, so we hard-code that partition rather than
        rely on a seed.

        ``seed`` is kept for API stability but unused; splitting is deterministic by dialogue
        order.
        """

        del seed  # deterministic; kept for API stability
        train_dialogues = self.dialogues[:1]
        val_dialogues = self.dialogues[1:2]
        test_dialogues = self.dialogues[2:]

        def flatten(ds: list[LoCoMoDialogue]) -> list[tuple[LoCoMoDialogue, LoCoMoQA]]:
            return [(d, q) for d in ds for q in d.qa]

        return flatten(train_dialogues), flatten(val_dialogues), flatten(test_dialogues)
