"""Per-character SceneIndex — CoSER given-circumstance retrieval.

For each turn we embed (recent-history-context + user_message) and retrieve
the most analogous past scene the character has been in. That scene gives us
all four GCA elements at once:
  • scenario      — element #1
  • our motivation — element #3 (this character's `thought` in the scene)
  • other characters' profiles — element #4 (other key_characters' `thought`s)
  • dialogues     — bonus reference for tone/format

Disk-cached just like UtteranceIndex.
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
from roleplaychatbotservice.personas.schema import BotPersona, Scene

logger = logging.getLogger(__name__)

CACHE_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / ".cache" / "scene"


@dataclass(frozen=True)
class RetrievedScene:
    scene: Scene
    score: float


def _doc_for(scene: Scene) -> str:
    parts = [scene.topic, scene.scenario]
    return "\n".join(p for p in parts if p)


class SceneIndex:
    def __init__(self, persona_id: str, scenes: tuple[Scene, ...], vectors: np.ndarray) -> None:
        self.persona_id = persona_id
        self._scenes = scenes
        if not scenes:
            self._index = None
            return
        self._index = faiss.IndexFlatIP(int(vectors.shape[1]))
        self._index.add(vectors)

    def search(self, query: str, embedder: QwenEmbedder, k: int = 1) -> list[RetrievedScene]:
        if self._index is None or not self._scenes:
            return []
        q = embedder.encode([query])
        scores, idx = self._index.search(q, min(k, len(self._scenes)))
        out: list[RetrievedScene] = []
        for s, i in zip(scores[0].tolist(), idx[0].tolist()):
            if i < 0:
                continue
            out.append(RetrievedScene(scene=self._scenes[i], score=float(s)))
        return out

    @classmethod
    def build(cls, persona: BotPersona, embedder: QwenEmbedder) -> "SceneIndex":
        scenes = persona.scenes
        if not scenes:
            return cls(persona.id, (), np.zeros((0, 1), dtype="float32"))

        cache_path = _cache_path(persona, embedder.model_id)
        vectors = _load_cached(cache_path, expected_n=len(scenes))
        if vectors is None:
            logger.info("embedding %d scenes for persona %s ...", len(scenes), persona.id)
            docs = [_doc_for(s) for s in scenes]
            vectors = embedder.encode(docs)
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, vectors)
        return cls(persona.id, scenes, vectors)


def _scene_fingerprint(persona: BotPersona) -> str:
    payload = json.dumps(
        [{"sc": s.scenario, "t": s.topic} for s in persona.scenes],
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _cache_path(persona: BotPersona, embedder_id: str) -> Path:
    fp = _scene_fingerprint(persona)
    embed_slug = embedder_id.replace("/", "_")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"{persona.id}__{embed_slug}__{fp}.npy"


def _load_cached(path: Path, expected_n: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        v = np.load(path)
    except Exception:
        return None
    if v.ndim != 2 or v.shape[0] != expected_n:
        return None
    return v.astype("float32")


def build_scene_indices(personas: list[BotPersona], embedder: QwenEmbedder) -> dict[str, SceneIndex]:
    out: dict[str, SceneIndex] = {}
    for p in personas:
        if not p.scenes:
            continue
        out[p.id] = SceneIndex.build(p, embedder)
    return out
