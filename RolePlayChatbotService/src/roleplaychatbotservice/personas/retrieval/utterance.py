"""Per-character FAISS index over a persona's rag_corpus.

ChatHaruhi-style: at each chat turn, embed the user's last message, retrieve
top-k chunks from this character's library, and inject them as few-shot in
the CHAI chat_history.

Disk cache: vectors are persisted under .cache/utt/<persona_id>__<hash>.npy
keyed by the corpus content + embedder model id, so process restarts skip
re-embedding (Qwen embedding is the slow part).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from roleplaychatbotservice.personas.embedder import QwenEmbedder
from roleplaychatbotservice.personas.schema import BotPersona, RagChunk

logger = logging.getLogger(__name__)

CACHE_ROOT = Path(__file__).resolve().parent.parent.parent.parent / ".cache" / "utt"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: RagChunk
    score: float


class UtteranceIndex:
    """One FAISS index per character. Construct via `UtteranceIndex.build`."""

    def __init__(self, persona_id: str, chunks: tuple[RagChunk, ...], vectors: np.ndarray) -> None:
        if not chunks:
            raise ValueError(f"persona {persona_id!r} has no rag_corpus chunks")
        self.persona_id = persona_id
        self._chunks = chunks
        dim = int(vectors.shape[1])
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)

    def search(self, query: str, embedder: QwenEmbedder, k: int = 4) -> list[RetrievedChunk]:
        if k <= 0 or not self._chunks:
            return []
        q = embedder.encode([query])
        scores, idx = self._index.search(q, min(k, len(self._chunks)))
        out: list[RetrievedChunk] = []
        for s, i in zip(scores[0].tolist(), idx[0].tolist()):
            if i < 0:
                continue
            out.append(RetrievedChunk(chunk=self._chunks[i], score=float(s)))
        return out

    @classmethod
    def build(cls, persona: BotPersona, embedder: QwenEmbedder) -> "UtteranceIndex":
        chunks = persona.rag_corpus
        cache_path = _cache_path(persona, embedder.model_id)
        vectors = _load_cached(cache_path, expected_n=len(chunks))
        if vectors is None:
            logger.info("embedding %d chunks for persona %s ...", len(chunks), persona.id)
            texts = [c.text for c in chunks]
            vectors = embedder.encode(texts)
            _save_cached(cache_path, vectors)
        return cls(persona.id, chunks, vectors)


def _corpus_fingerprint(persona: BotPersona) -> str:
    payload = json.dumps(
        [{"t": c.text, "k": c.kind, "s": c.source} for c in persona.rag_corpus],
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _cache_path(persona: BotPersona, embedder_id: str) -> Path:
    fp = _corpus_fingerprint(persona)
    embed_slug = embedder_id.replace("/", "_")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"{persona.id}__{embed_slug}__{fp}.npy"


def _load_cached(path: Path, expected_n: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        v = np.load(path)
    except Exception as exc:
        logger.warning("could not load cache %s: %s", path, exc)
        return None
    if v.ndim != 2 or v.shape[0] != expected_n:
        logger.info("cache shape mismatch for %s, will re-embed", path.name)
        return None
    logger.debug("loaded %d vectors from cache %s", v.shape[0], path.name)
    return v.astype("float32")


def _save_cached(path: Path, vectors: np.ndarray) -> None:
    np.save(path, vectors)
    logger.debug("cached %d vectors → %s", vectors.shape[0], path)


def build_indices(personas: list[BotPersona], embedder: QwenEmbedder) -> dict[str, UtteranceIndex]:
    """Build one UtteranceIndex per persona at startup."""
    indices: dict[str, UtteranceIndex] = {}
    for p in personas:
        if not p.rag_corpus:
            logger.warning("persona %s has no rag_corpus, skipping index", p.id)
            continue
        indices[p.id] = UtteranceIndex.build(p, embedder)
    return indices
