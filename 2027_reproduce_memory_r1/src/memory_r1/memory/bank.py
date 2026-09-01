"""MemoryBank: the external store the Memory Manager reads and writes.

Design choices matching the paper:

- Per-speaker banks: the paper's Answer Agent prompt (Figure 11) separates "Memories for user X" and
  "Memories for user Y", so we keep a separate list per speaker inside a single MemoryBank.
- Integer string IDs monotonically assigned per speaker, matching the ADD examples in Figure 9
  ("id": "0", "id": "1", ...). This is what the Memory Manager prompt expects.
- Optional timestamp on each entry — the Answer Agent prompt relies on timestamps for temporal
  reasoning ("May special attention to the timestamps to determine the answer").
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MemoryEntry:
    """One row in the memory bank."""

    id: str
    text: str
    speaker: str
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=str(d["id"]),
            text=str(d["text"]),
            speaker=str(d.get("speaker", "")),
            timestamp=d.get("timestamp"),
            metadata=dict(d.get("metadata", {})),
        )


class MemoryBank:
    """Speaker-keyed collection of MemoryEntry with monotonic per-speaker IDs.

    The ID sequence is per-speaker so the Manager prompt sees a small, stable range of IDs to
    reference (0, 1, 2, ...). Deletes leave the counter monotonic to avoid ID reuse.
    """

    def __init__(self) -> None:
        self._entries: dict[str, list[MemoryEntry]] = defaultdict(list)
        self._next_id: dict[str, int] = defaultdict(int)

    # --------------------------------------------------------------------- basic access

    def speakers(self) -> list[str]:
        return sorted(self._entries.keys())

    def entries_of(self, speaker: str) -> list[MemoryEntry]:
        return list(self._entries.get(speaker, []))

    def all_entries(self) -> list[MemoryEntry]:
        out: list[MemoryEntry] = []
        for speaker in self.speakers():
            out.extend(self._entries[speaker])
        return out

    def __len__(self) -> int:
        return sum(len(v) for v in self._entries.values())

    def __iter__(self) -> Iterator[MemoryEntry]:
        return iter(self.all_entries())

    def find(self, speaker: str, entry_id: str) -> MemoryEntry | None:
        for e in self._entries.get(speaker, []):
            if e.id == entry_id:
                return e
        return None

    # --------------------------------------------------------------------- mutations

    def add(self, speaker: str, text: str, timestamp: str | None = None) -> MemoryEntry:
        """ADD op — allocate a fresh per-speaker ID."""
        eid = str(self._next_id[speaker])
        self._next_id[speaker] += 1
        entry = MemoryEntry(id=eid, text=text, speaker=speaker, timestamp=timestamp)
        self._entries[speaker].append(entry)
        return entry

    def update(self, speaker: str, entry_id: str, new_text: str) -> bool:
        """UPDATE op — mutate text of existing entry. Returns True on success."""
        for e in self._entries.get(speaker, []):
            if e.id == entry_id:
                e.text = new_text
                return True
        return False

    def delete(self, speaker: str, entry_id: str) -> bool:
        """DELETE op — remove entry (ID is NOT recycled)."""
        bucket = self._entries.get(speaker, [])
        for i, e in enumerate(bucket):
            if e.id == entry_id:
                del bucket[i]
                return True
        return False

    # --------------------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": {k: [e.to_dict() for e in v] for k, v in self._entries.items()},
            "next_id": dict(self._next_id),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryBank":
        bank = cls()
        for speaker, items in d.get("entries", {}).items():
            bank._entries[speaker] = [MemoryEntry.from_dict(x) for x in items]
        for speaker, next_id in d.get("next_id", {}).items():
            bank._next_id[speaker] = int(next_id)
        return bank

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "MemoryBank":
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # --------------------------------------------------------------------- prompt-shaped views

    def as_prompt_list(self, speaker: str) -> list[dict[str, str]]:
        """The ``[{"id", "text"}, ...]`` shape the Memory Manager prompt expects."""
        return [{"id": e.id, "text": e.text} for e in self._entries.get(speaker, [])]

    def snapshot(self) -> "MemoryBank":
        return MemoryBank.from_dict(self.to_dict())
