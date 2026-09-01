"""Memory Manager prompt — paper Figures 9 (Part 1) and 10 (Part 2), verbatim.

The Memory Manager is instructed to emit a JSON object of the form::

    {
      "memory": [
        {"id": "0", "text": "...", "event": "ADD" | "UPDATE" | "DELETE" | "NONE",
         "old_memory": "... (only for UPDATE) ..."}
      ]
    }

We keep the "NONE" spelling from the paper for the no-op event even though the paper text refers to
this operation as ``NOOP`` — the prompt in Figure 10 uses ``NONE``.
"""

from __future__ import annotations

import json
from typing import Iterable, Mapping

MEMORY_MANAGER_SYSTEM = """\
You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update
the memory, (3) delete from the memory, and (4) no change.

Based on the above four operations, the memory will change.

Compare newly retrieved facts with the existing memory. For each new fact,
decide whether to:
- ADD: Add it to the memory as a new element
- UPDATE: Update an existing memory element
- DELETE: Delete an existing memory element
- NONE: Make no change (if the fact is already present or irrelevant)

1. **Add**: If the retrieved facts contain new information not present
in the memory, then you have to add it by generating a new ID in the id field.

- Example:
    Old Memory:
        [
            {"id" : "0", "text" : "User is a software engineer"}
        ]
    Retrieved facts: ["Name is John"]

    New Memory:
        {
            "memory" : [
                {"id" : "0", "text" : "User is a software engineer", "event" : "NONE"},
                {"id" : "1", "text" : "Name is John", "event" : "ADD"}
            ]
        }

2. **Update**: If the retrieved facts contain information that is already
present in the memory but the information is totally different, then
you have to update it.

If the retrieved fact contains information that conveys the same thing as
the memory, keep the version with more detail.

Example (a) -- if the memory contains "User likes to play cricket" and the
retrieved fact is "Loves to play cricket with friends", then update the
memory with the retrieved fact.

Example (b) -- if the memory contains "Likes cheese pizza" and the
retrieved fact is "Loves cheese pizza", then do NOT update it because they
convey the same information.

Important: When updating, keep the same ID and preserve old_memory.

- Example:
    Old Memory:
        [
            {"id" : "0", "text" : "I really like cheese pizza"},
            {"id" : "2", "text" : "User likes to play cricket"}
        ]
    Retrieved facts: ["Loves chicken pizza", "Loves to play cricket with friends"]

    New Memory:
        {
            "memory" : [
                {"id" : "0", "text" : "Loves cheese and chicken pizza", "event" : "UPDATE",
                    "old_memory" : "I really like cheese pizza"},
                {"id" : "2", "text" : "Loves to play cricket with friends", "event" : "UPDATE",
                    "old_memory" : "User likes to play cricket"}
            ]
        }

3. **Delete**: If the retrieved facts contain information that contradicts
the information in the memory, delete it. When deleting, return the same IDs -- do not generate new IDs.

- Example:
    Old Memory:
        [
            {"id" : "1", "text" : "Loves cheese pizza"}
        ]
    Retrieved facts: ["Dislikes cheese pizza"]

    New Memory:
        {
            "memory" : [
                {"id" : "1", "text" : "Loves cheese pizza", "event" : "DELETE"}
            ]
        }

4. **No Change**: If the retrieved facts are already present, make no change.

- Example:
    Old Memory:
        [
            {"id" : "0", "text" : "Name is John"}
        ]
    Retrieved facts: ["Name is John"]

    New Memory:
        {
            "memory" : [
                {"id" : "0", "text" : "Name is John", "event" : "NONE"}
            ]
        }

Respond with a single JSON object with a top-level "memory" key. Do not add any prose
outside the JSON.
"""


def _format_memory(old_memory: Iterable[Mapping[str, str]]) -> str:
    """Format the old memory bank as a JSON list of ``{"id", "text"}`` dicts."""

    items = [{"id": str(m["id"]), "text": str(m["text"])} for m in old_memory]
    return json.dumps(items, indent=4, ensure_ascii=False)


def build_manager_prompt(
    retrieved_facts: list[str],
    old_memory: list[Mapping[str, str]],
) -> str:
    """Build the user-turn prompt combining old memory bank and newly retrieved facts."""

    formatted_old = _format_memory(old_memory)
    formatted_facts = json.dumps(retrieved_facts, ensure_ascii=False)
    return (
        f"Old Memory:\n{formatted_old}\n\n"
        f"Retrieved facts: {formatted_facts}\n\n"
        "Return the New Memory JSON."
    )
