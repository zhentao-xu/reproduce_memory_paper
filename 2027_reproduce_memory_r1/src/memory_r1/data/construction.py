"""Data construction pipelines from paper Appendix B.2.

Algorithm 1 (Memory Manager training data)
------------------------------------------
For each dialogue ``d`` and each turn ``t`` in ``d``:
  1. Build a *temporal memory bank* from the previous 24 turns (paper says "up to 50" as an upper
     bound, "24 turns" in Appendix B.2). We take the paper's exact 24.
  2. Fuse (i) that temporal bank, (ii) the current turn ``t``, and (iii) any QA pair linked to
     ``t`` as a single training tuple.

Algorithm 2 (Answer Agent training data)
----------------------------------------
For each question ``q`` in ``d``:
  1. Use the Manager to build an up-to-date memory bank over ``d``.
  2. Retrieve top-30 candidate memories per participant (60 total) from the bank using ``q``.
  3. Store ``(question, retrieved_60, gold)`` as a training tuple.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track

from memory_r1.agents.extractor import FactExtractor
from memory_r1.agents.llm_backend import LLMBackend
from memory_r1.agents.memory_manager import MemoryManager
from memory_r1.data.locomo import LoCoMoDialogue, LoCoMoLoader, LoCoMoTurn
from memory_r1.memory.bank import MemoryBank
from memory_r1.memory.retrieval import DenseRetriever

console = Console()


# --------------------------------------------------------------------------- helpers


def _turn_qa_map(dialogue: LoCoMoDialogue) -> dict[str, list[dict[str, str]]]:
    """Map ``dia_id`` -> QA items whose evidence cites that dia_id.

    LoCoMo QA items have an ``evidence`` field listing the ``dia_id`` values whose contents support
    the answer. We use this to attach a QA pair to the *last* turn that produces enough evidence,
    so the Memory Manager gets signal from question difficulty tied to the exact turn.
    """

    m: dict[str, list[dict[str, str]]] = {}
    for q in dialogue.qa:
        for ev in q.evidence:
            m.setdefault(str(ev), []).append({"question": q.question, "answer": q.answer, "category": str(q.category)})
    return m


# --------------------------------------------------------------------------- Algorithm 1


def build_manager_dataset(
    locomo_path: Path,
    out_path: Path,
    lookback_turns: int = 24,
    extractor_backend: LLMBackend | None = None,
    max_dialogues: int | None = None,
) -> None:
    """Build (dialogue_turn, temporal_memory_bank, QA) training tuples.

    Parameters
    ----------
    locomo_path : Path
        Path to ``locomo10.json``.
    out_path : Path
        Where to write the JSONL. One line per turn that has extractable facts.
    lookback_turns : int
        Number of previous turns used to build the temporal memory bank. Default 24 (paper).
    extractor_backend : LLMBackend | None
        Backend used by the fact extractor. If ``None``, we use a heuristic (split utterance into
        sentences) so this script works offline for smoke tests. In practice you'll want an OpenAI
        backend targeting ``gpt-4o-mini``.
    max_dialogues : int | None
        Optional cap for quick iteration.
    """

    loader = LoCoMoLoader(locomo_path, exclude_adversarial=True)
    dialogues = loader.dialogues if max_dialogues is None else loader.dialogues[:max_dialogues]

    extractor = _make_extractor(extractor_backend)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for dialogue in track(dialogues, description="Algorithm 1: manager data"):
            qa_map = _turn_qa_map(dialogue)
            for i, turn in enumerate(dialogue.turns):
                start = max(0, i - lookback_turns)
                context_turns = dialogue.turns[start:i]

                # Build the temporal memory bank *for the previous 24 turns* using the extractor.
                temporal_bank = MemoryBank()
                for ctx in context_turns:
                    facts_ctx = extractor(ctx.speaker, ctx.text, ctx.timestamp)
                    for fact in facts_ctx:
                        temporal_bank.add(ctx.speaker, fact, timestamp=ctx.timestamp)

                # Extract facts from the current turn (the Manager's rollout target).
                facts_now = extractor(turn.speaker, turn.text, turn.timestamp)
                if not facts_now:
                    continue

                qa_items = qa_map.get(turn.dia_id or "", [])

                record = {
                    "dialogue_id": dialogue.dialogue_id,
                    "turn_index": turn.turn_index_global,
                    "session_id": turn.session_id,
                    "dia_id": turn.dia_id,
                    "speaker": turn.speaker,
                    "timestamp": turn.timestamp,
                    "utterance": turn.text,
                    "extracted_facts": facts_now,
                    "temporal_memory_bank": temporal_bank.to_dict(),
                    "linked_qa": qa_items,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

    console.print(f"[green]Wrote {n_written} manager tuples → {out_path}[/green]")


# --------------------------------------------------------------------------- Algorithm 2


def build_answer_dataset(
    locomo_path: Path,
    out_path: Path,
    top_k_per_speaker: int = 30,
    manager_backend: LLMBackend | None = None,
    extractor_backend: LLMBackend | None = None,
    retriever: DenseRetriever | None = None,
    max_dialogues: int | None = None,
    chunking: str = "fact",
) -> None:
    """Build (question, retrieved_60, gold) tuples for the Answer Agent.

    ``manager_backend`` is used to run the Memory Manager over each dialogue to build the memory
    bank that gets queried. If ``None``, we bypass the Manager and add every extracted fact
    directly (i.e. an "always ADD" baseline manager). This is fine for pre-RL data construction —
    the Answer Agent is trained on retrieval-over-facts, not on Manager quality.

    ``chunking`` controls how utterances become memory entries:

    - ``"fact"`` (default, paper): run the fact extractor per turn; each atomic fact is one
      memory entry. Faithful to Algorithm 2 but Q and A are frequently split across entries,
      so retrieval can surface one without the other.
    - ``"turn_pair"``: skip the extractor and Manager entirely. Walk the dialogue as
      OVERLAPPING pairs of consecutive turns (sliding window, stride 1) so both the A→B and
      B→A adjacencies are indexed; each pair becomes ONE memory entry with text
      ``"{sp_a}: {txt_a}\\n{sp_b}: {txt_b}"``. The pair is registered under BOTH speakers'
      buckets so per-speaker top-K retrieval finds it regardless of who opened the exchange.
      Useful for Answer-Agent GRPO/DPO training where you want the retrieved context to keep
      question+response coherent.
    """

    if chunking not in {"fact", "turn_pair"}:
        raise ValueError(f"chunking must be 'fact' or 'turn_pair', got {chunking!r}")

    loader = LoCoMoLoader(locomo_path, exclude_adversarial=True)
    dialogues = loader.dialogues if max_dialogues is None else loader.dialogues[:max_dialogues]

    if retriever is None:
        retriever = DenseRetriever()

    extractor = _make_extractor(extractor_backend) if chunking == "fact" else None
    manager = MemoryManager(manager_backend, retriever=retriever) if (chunking == "fact" and manager_backend) else None

    from memory_r1.utils import logger

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for di, dialogue in enumerate(dialogues, 1):
            logger.info("📚 dialogue {}/{} id={} chunking={} — building bank from {} turns",
                        di, len(dialogues), dialogue.dialogue_id, chunking, len(dialogue.turns))
            if chunking == "turn_pair":
                bank = _construct_bank_pair_chunks(dialogue)
            else:
                bank = _construct_bank_for_dialogue(
                    dialogue=dialogue, extractor=extractor, manager=manager
                )
            logger.info("🗃️  bank built: {} memories across {} speakers",
                        len(bank), len(bank.speakers()))
            logger.info("🔍 retrieving top-{}/speaker for {} questions (batched)...",
                        top_k_per_speaker, len(dialogue.qa))

            # Batched retrieval: encode bank once, encode all queries once, sort per-question in
            # NumPy. Previously we called search_by_speaker inside the loop, which re-encoded the
            # entire bank on every question (~150× redundant work for a 150-question dialogue).
            all_retrieved = retriever.search_by_speaker_batch(
                bank, [q.question for q in dialogue.qa], top_k_per_speaker=top_k_per_speaker
            )

            for qi, (q, retrieved) in enumerate(zip(dialogue.qa, all_retrieved, strict=True), 1):
                mem_by_speaker: dict[str, list[dict[str, str]]] = {}
                for sp, hits in retrieved.items():
                    mem_by_speaker[sp] = [
                        {
                            "id": h.entry_id,
                            "text": h.text,
                            "timestamp": h.timestamp or "",
                            "score": f"{h.score:.4f}",
                        }
                        for h in hits
                    ]

                record = {
                    "dialogue_id": dialogue.dialogue_id,
                    "question": q.question,
                    "gold_answer": q.answer,
                    "category": q.category,
                    "category_name": q.category_name,
                    "retrieved": mem_by_speaker,
                    "bank_snapshot": bank.to_dict(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1
                if qi % 10 == 0 or qi == len(dialogue.qa):
                    logger.info("   ↳ Q {}/{}: '{:.60s}...' gold={!r}",
                                qi, len(dialogue.qa), q.question, q.answer[:40])

    logger.success("🏁 wrote {} answer tuples → {}", n_written, out_path)


# --------------------------------------------------------------------------- shared


def _construct_bank_for_dialogue(
    dialogue: LoCoMoDialogue,
    extractor,
    manager: MemoryManager | None,
) -> MemoryBank:
    """Run Algorithm 3 over one dialogue and return the final MemoryBank."""

    bank = MemoryBank()
    for turn in dialogue.turns:
        facts = extractor(turn.speaker, turn.text, turn.timestamp)
        if not facts:
            continue
        if manager is None:
            for fact in facts:
                bank.add(turn.speaker, fact, timestamp=turn.timestamp)
        else:
            manager.step(bank, speaker=turn.speaker, facts=facts, turn_timestamp=turn.timestamp)
    return bank


def _construct_bank_pair_chunks(dialogue: LoCoMoDialogue) -> MemoryBank:
    """Chunk the dialogue as OVERLAPPING pairs of consecutive turns (sliding window, stride 1).

    For turns ``[t0, t1, t2, t3, ...]`` we emit entries for ``(t0,t1)``, ``(t1,t2)``,
    ``(t2,t3)``, ... — so both the A→B and the B→A adjacencies are indexed. Each pair becomes
    one memory entry with the two utterances concatenated (speaker-prefixed) and is registered
    under BOTH speakers' buckets so per-speaker top-K retrieval finds it regardless of which
    speaker opened the pair. A single-turn dialogue is stored as a lone singleton.
    """

    bank = MemoryBank()
    turns = dialogue.turns
    if not turns:
        return bank
    if len(turns) == 1:
        a = turns[0]
        bank.add(a.speaker, f"{a.speaker}: {a.text}", timestamp=a.timestamp)
        return bank
    for i in range(len(turns) - 1):
        a, b = turns[i], turns[i + 1]
        text = f"{a.speaker}: {a.text}\n{b.speaker}: {b.text}"
        ts = a.timestamp
        for sp in {a.speaker, b.speaker}:
            bank.add(sp, text, timestamp=ts)
    return bank


def _make_extractor(backend: LLMBackend | None):
    """Return a callable ``(speaker, utterance, ts) -> list[str]``.

    If a real LLM backend is provided, uses ``FactExtractor``. Otherwise falls back to a naive
    sentence-splitter — good enough for smoke tests and CI, but you'll want GPT-4o-mini for the
    real experiments (matches the paper).
    """

    if backend is None:
        import re

        def _heuristic(speaker: str, text: str, ts: str | None) -> list[str]:
            text = text.strip()
            if not text:
                return []
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            return [f"{speaker} said: {s}" for s in sentences if len(s) > 3]

        return _heuristic

    fx = FactExtractor(backend)

    def _real(speaker: str, text: str, ts: str | None) -> list[str]:
        return fx.extract(speaker, text, ts).facts

    return _real


# --------------------------------------------------------------------------- split


def write_locomo_splits(
    locomo_path: Path,
    out_dir: Path,
    seed: int = 42,
) -> None:
    """Persist the paper's 1:1:8 QA split as JSONL for reproducibility."""

    loader = LoCoMoLoader(locomo_path, exclude_adversarial=True)
    train, val, test = loader.qa_split_152_81_1307(seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"locomo_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for dialogue, qa in split:
                f.write(json.dumps({
                    "dialogue_id": dialogue.dialogue_id,
                    "question": qa.question,
                    "gold_answer": qa.answer,
                    "category": qa.category,
                    "category_name": qa.category_name,
                }, ensure_ascii=False) + "\n")
        console.print(f"[green]{name}: {len(split)} QA → {path}[/green]")
