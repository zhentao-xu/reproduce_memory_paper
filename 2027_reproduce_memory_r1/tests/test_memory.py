"""Smoke tests for the memory-bank data structures and Manager output parsing."""

from __future__ import annotations

import json

from memory_r1.memory.bank import MemoryBank
from memory_r1.memory.operations import ManagerOp, apply_manager_output, parse_manager_output


def test_bank_add_update_delete():
    bank = MemoryBank()
    e1 = bank.add("alice", "Alice loves pizza")
    e2 = bank.add("alice", "Alice is a runner")
    assert e1.id == "0" and e2.id == "1"
    assert len(bank) == 2

    assert bank.update("alice", "0", "Alice loves pizza and pasta")
    assert bank.find("alice", "0").text.endswith("pasta")

    assert bank.delete("alice", "1")
    assert len(bank) == 1
    # ID never re-used.
    e3 = bank.add("alice", "Alice climbs")
    assert e3.id == "2"


def test_parse_manager_output_add_update_delete_none():
    payload = {
        "memory": [
            {"id": "0", "text": "Alice is a software engineer", "event": "NONE"},
            {"id": "1", "text": "Name is John", "event": "ADD"},
            {"id": "2", "text": "Loves cheese pizza", "event": "UPDATE",
             "old_memory": "Likes pizza"},
            {"id": "3", "text": "Old thing", "event": "DELETE"},
        ]
    }
    parsed = parse_manager_output(json.dumps(payload))
    assert [a.event for a in parsed.actions] == [
        ManagerOp.NOOP, ManagerOp.ADD, ManagerOp.UPDATE, ManagerOp.DELETE
    ]


def test_apply_manager_output_updates_bank():
    bank = MemoryBank()
    bank.add("alice", "Alice is a software engineer")  # id 0

    payload = {
        "memory": [
            {"id": "0", "text": "Alice is a senior software engineer", "event": "UPDATE",
             "old_memory": "Alice is a software engineer"},
            {"id": "1", "text": "Alice loves pizza", "event": "ADD"},
        ]
    }
    parsed = parse_manager_output(json.dumps(payload))
    counts = apply_manager_output(
        speaker="alice",
        old_memory_view=bank.as_prompt_list("alice"),
        output=parsed,
        bank=bank,
    )
    assert counts["ADD"] == 1 and counts["UPDATE"] == 1
    assert any("senior" in e.text for e in bank.entries_of("alice"))
    assert any("pizza" in e.text for e in bank.entries_of("alice"))


def test_bank_roundtrip():
    bank = MemoryBank()
    bank.add("alice", "A")
    bank.add("bob", "B", timestamp="2026-01-01")
    d = bank.to_dict()
    bank2 = MemoryBank.from_dict(d)
    assert len(bank2) == 2
    assert bank2.find("bob", "0").timestamp == "2026-01-01"
