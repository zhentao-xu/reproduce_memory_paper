"""Parse the Memory Manager's JSON output and apply it to a MemoryBank.

The paper's Memory Manager prompt (Figures 9-10) instructs the model to emit::

    {
      "memory": [
        {"id": "<int>", "text": "<new content>", "event": "ADD"|"UPDATE"|"DELETE"|"NONE",
         "old_memory": "..." (UPDATE only)}
      ]
    }

We keep the paper's ``NONE`` spelling for the no-op event even though it corresponds to ``NOOP`` in
the algorithm text (Algorithm 5).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ManagerOp(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NONE"


@dataclass
class ManagerAction:
    event: ManagerOp
    text: str
    entry_id: str | None = None  # required for UPDATE/DELETE/NOOP; None → new for ADD
    old_memory: str | None = None  # UPDATE only, for auditability


@dataclass
class ManagerOutput:
    actions: list[ManagerAction]
    raw: str  # the original text produced by the model


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_block(raw: str) -> str:
    """Extract the first JSON object from the model output, tolerating leading/trailing prose."""

    raw = raw.strip()
    if raw.startswith("```"):
        # Handle ```json ... ``` fences.
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    m = _JSON_OBJ_RE.search(raw)
    if not m:
        raise ValueError("No JSON object found in Manager output.")
    return m.group(0)


def parse_manager_output(raw: str) -> ManagerOutput:
    """Parse the Manager's raw text into a structured ``ManagerOutput``.

    Malformed outputs raise ``ValueError``; the RL reward will interpret that as a failed rollout
    (Answer Agent falls back to the un-updated bank).
    """

    payload_text = _extract_json_block(raw)
    payload = json.loads(payload_text)
    if "memory" not in payload:
        raise ValueError("Manager output missing top-level 'memory' key.")

    actions: list[ManagerAction] = []
    for item in payload["memory"]:
        event_raw = str(item.get("event", "")).upper()
        # Tolerate NOOP as a synonym of NONE, since the paper text uses NOOP.
        if event_raw in ("NOOP", "NONE"):
            event = ManagerOp.NOOP
        else:
            try:
                event = ManagerOp(event_raw)
            except ValueError as e:
                raise ValueError(f"Unknown event '{event_raw}'.") from e
        actions.append(
            ManagerAction(
                event=event,
                text=str(item.get("text", "")),
                entry_id=str(item["id"]) if "id" in item and item["id"] is not None else None,
                old_memory=item.get("old_memory"),
            )
        )
    return ManagerOutput(actions=actions, raw=raw)


def apply_manager_output(
    speaker: str,
    old_memory_view: list[dict[str, str]],
    output: ManagerOutput,
    bank,  # MemoryBank, avoid circular import
    turn_timestamp: str | None = None,
) -> dict[str, int]:
    """Apply ``output.actions`` to ``bank`` for ``speaker``. Returns per-op counts.

    ``old_memory_view`` is the ``as_prompt_list(speaker)`` snapshot passed to the model; we use it
    to sanity-check that referenced IDs actually exist before mutating.
    """

    known_ids = {m["id"] for m in old_memory_view}
    counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}

    for action in output.actions:
        if action.event is ManagerOp.ADD:
            bank.add(speaker=speaker, text=action.text, timestamp=turn_timestamp)
            counts["ADD"] += 1
        elif action.event is ManagerOp.UPDATE:
            if action.entry_id is not None and action.entry_id in known_ids:
                bank.update(speaker=speaker, entry_id=action.entry_id, new_text=action.text)
                counts["UPDATE"] += 1
        elif action.event is ManagerOp.DELETE:
            if action.entry_id is not None and action.entry_id in known_ids:
                bank.delete(speaker=speaker, entry_id=action.entry_id)
                counts["DELETE"] += 1
        elif action.event is ManagerOp.NOOP:
            counts["NOOP"] += 1
    return counts


def dumps_manager_output(actions: list[ManagerAction]) -> str:
    """Reference formatter: turn a list of actions back into the JSON string that the prompt asks
    the model to produce. Useful for building supervised targets in Memory-SFT baselines."""

    payload: dict[str, list[dict[str, Any]]] = {"memory": []}
    for a in actions:
        item: dict[str, Any] = {
            "id": a.entry_id if a.entry_id is not None else "",
            "text": a.text,
            "event": a.event.value,
        }
        if a.event is ManagerOp.UPDATE and a.old_memory is not None:
            item["old_memory"] = a.old_memory
        payload["memory"].append(item)
    return json.dumps(payload, ensure_ascii=False)
