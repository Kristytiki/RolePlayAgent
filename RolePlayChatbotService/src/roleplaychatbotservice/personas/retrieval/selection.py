"""Global FAISS index over persona cards — RAG layer 1.

User types 'a brilliant overachiever' → top-k personas. One vector per persona,
embedded from name + tagline + tags + a profile excerpt.
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
from roleplaychatbotservice.personas.schema import BotPersona

logger = logging.getLogger(__name__)

CACHE_ROOT = Path(__file__).resolve().parent.parent.parent.parent / ".cache" / "selection"


@dataclass(frozen=True)
class RetrievedPersona:
    persona: BotPersona
    score: float


def _doc_for(p: BotPersona) -> str:
    profile_head = (p.profile_markdown or "")[:600]
    return "\n".join([
        f"Name: {p.name}",
        f"Source: {p.source}",
        f"Tagline: {p.tagline}",
        f"Tags: {', '.join(p.tags)}",
        f"Profile: {profile_head}",
    ])


def _fingerprint(personas: list[BotPersona]) -> str:
    payload = json.dumps(
        [{"id": p.id, "doc": _doc_for(p)} for p in personas],
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class PersonaSelectionIndex:
    def __init__(self, personas: list[BotPersona], embedder: QwenEmbedder) -> None:
        self._personas = list(personas)
        self._embedder = embedder
        if not self._personas:
            raise ValueError("PersonaSelectionIndex requires at least one persona")
        vectors = self._build_vectors()
        dim = int(vectors.shape[1])
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)

    def _build_vectors(self) -> np.ndarray:
        fp = _fingerprint(self._personas)
        embed_slug = self._embedder.model_id.replace("/", "_")
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_ROOT / f"selection__{embed_slug}__{fp}.npy"
        if cache_path.exists():
            try:
                v = np.load(cache_path)
                if v.shape[0] == len(self._personas):
                    logger.info("loaded selection index from cache (%d vectors)", v.shape[0])
                    return v.astype("float32")
            except Exception:
                pass
        logger.info("building selection index for %d personas ...", len(self._personas))
        docs = [_doc_for(p) for p in self._personas]
        v = self._embedder.encode(docs)
        np.save(cache_path, v)
        return v

    def search(self, query: str, k: int = 5) -> list[RetrievedPersona]:
        q = self._embedder.encode([query])
        scores, idx = self._index.search(q, min(k, len(self._personas)))
        out: list[RetrievedPersona] = []
        for s, i in zip(scores[0].tolist(), idx[0].tolist()):
            if i < 0:
                continue
            out.append(RetrievedPersona(persona=self._personas[i], score=float(s)))
        return out
