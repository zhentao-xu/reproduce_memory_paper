"""Dense retriever used by (1) Memory Manager to search the current bank for related facts and
(2) Answer Agent training data construction, which retrieves top-30 memories per participant.

We use ``sentence-transformers`` with cosine similarity via FAISS. The paper doesn't fix a specific
encoder; ``all-MiniLM-L6-v2`` is the widely-used default for RAG baselines like Mem0 (Chhikara et
al., 2025), which Memory-R1 explicitly compares against, so we default to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from memory_r1.memory.bank import MemoryBank, MemoryEntry


DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

# Encoders that need special query / passage prefixes for their contrastive training regime.
# ``e5-*`` models (Wang et al., "Text Embeddings by Weakly-Supervised Contrastive Pre-training",
# arXiv:2212.03533) were trained with "query: ..." for the anchor side and "passage: ..." for the
# document side; skipping these prefixes silently degrades retrieval quality by 5-10 F1.
_QUERY_PREFIX_ENCODERS: dict[str, tuple[str, str]] = {
    # (query_prefix, passage_prefix)
    "intfloat/e5-large-v2": ("query: ", "passage: "),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    "intfloat/e5-small-v2": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "BAAI/bge-large-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
}


@dataclass
class Retrieved:
    """One hit from the retriever."""

    entry_id: str
    speaker: str
    text: str
    score: float
    timestamp: str | None = None


class DenseRetriever:
    """Cosine-similarity retriever backed by a sentence-transformer encoder.

    We keep the retriever *stateless w.r.t. the memory bank*: every ``search`` call re-encodes the
    query and computes similarities against the current bank contents. This is a small ergonomic
    hit at Manager-training time (< 1s for <100 entries) but keeps semantics simple: no stale index
    after ADD/UPDATE/DELETE. If speed matters, use ``build_index`` + ``search_index``.
    """

    def __init__(self, encoder_name: str = DEFAULT_ENCODER, device: str | None = None) -> None:
        # Deferred import so unit tests that don't need retrieval don't pay the load cost.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(encoder_name, device=device)
        self._encoder_name = encoder_name
        self._query_prefix, self._passage_prefix = _QUERY_PREFIX_ENCODERS.get(encoder_name, ("", ""))

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        """Encode a batch of strings.

        Set ``is_query=True`` when encoding the QUERY side (the question the user is asking) so
        query-only prefixes for e5/bge are applied; leave as False for document-side encoding.
        """

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self._query_prefix if is_query else self._passage_prefix
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]
        emb = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return emb.astype(np.float32)

    # ---------------------------------------------------------- Manager-time (small bank)

    def search_bank(
        self,
        bank: "MemoryBank",
        query: str,
        top_k: int,
        speaker: str | None = None,
    ) -> list[Retrieved]:
        """Return the top-``top_k`` entries most similar to ``query`` from ``bank``.

        If ``speaker`` is provided, restrict to that speaker's memories. Used for two purposes:

        - Memory-Manager rollout: retrieve the top-K existing entries that are relevant to the
          incoming facts (so the Manager sees a small ``old_memory`` context, not the whole bank).
        - Answer-Agent data construction: retrieve top-30 per speaker for each question (Algo 2).
        """

        entries: list[MemoryEntry] = (
            bank.entries_of(speaker) if speaker is not None else bank.all_entries()
        )
        if not entries:
            return []
        emb = self.encode([e.text for e in entries], is_query=False)
        q = self.encode([query], is_query=True)[0]
        sims = emb @ q  # cosine because vectors are unit-normalized
        order = np.argsort(-sims)[: top_k]
        return [
            Retrieved(
                entry_id=entries[i].id,
                speaker=entries[i].speaker,
                text=entries[i].text,
                score=float(sims[i]),
                timestamp=entries[i].timestamp,
            )
            for i in order
        ]

    def search_by_speaker(
        self,
        bank: "MemoryBank",
        query: str,
        top_k_per_speaker: int,
    ) -> dict[str, list[Retrieved]]:
        """Answer-Agent's canonical retrieval: top-K per speaker.

        The paper says: "60 candidate memories are retrieved for each question via similarity-based
        RAG" and Algorithm 2 clarifies this is top-30 per participant, both speakers.
        """

        return {sp: self.search_bank(bank, query, top_k_per_speaker, speaker=sp) for sp in bank.speakers()}

    def search_by_speaker_batch(
        self,
        bank: "MemoryBank",
        queries: list[str],
        top_k_per_speaker: int,
    ) -> list[dict[str, list["Retrieved"]]]:
        """Vectorized ``search_by_speaker`` for a fixed bank + many queries.

        The single-query API re-encodes the bank on every call — fine when the bank changes
        turn-by-turn (Manager rollouts) but wasteful for Answer-Agent data construction where the
        bank is built once and then queried by all ~150 questions in a dialogue. This method
        encodes the bank ONCE per speaker and all queries ONCE as a batch, then does the sort
        per (query, speaker) purely in NumPy. ~50-100× faster than looping ``search_by_speaker``.
        """

        speakers = bank.speakers()
        if not queries:
            return []

        # Encode each speaker's bank ONCE (was ``len(queries)`` times before).
        per_speaker_entries: dict[str, list["MemoryEntry"]] = {}
        per_speaker_emb: dict[str, np.ndarray] = {}
        for sp in speakers:
            entries = bank.entries_of(sp)
            per_speaker_entries[sp] = entries
            per_speaker_emb[sp] = (
                self.encode([e.text for e in entries], is_query=False)
                if entries
                else np.zeros((0, self.dim), dtype=np.float32)
            )

        # Encode all queries in one shot so sentence-transformers can batch the forward pass.
        q_emb = self.encode(queries, is_query=True)  # shape: (n_queries, dim)

        results: list[dict[str, list[Retrieved]]] = []
        for qi in range(len(queries)):
            per_q: dict[str, list[Retrieved]] = {}
            for sp in speakers:
                entries = per_speaker_entries[sp]
                if not entries:
                    per_q[sp] = []
                    continue
                sims = per_speaker_emb[sp] @ q_emb[qi]  # (n_entries,)
                order = np.argsort(-sims)[:top_k_per_speaker]
                per_q[sp] = [
                    Retrieved(
                        entry_id=entries[i].id,
                        speaker=entries[i].speaker,
                        text=entries[i].text,
                        score=float(sims[i]),
                        timestamp=entries[i].timestamp,
                    )
                    for i in order
                ]
            results.append(per_q)
        return results
