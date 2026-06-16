"""Wrapper around sentence-transformers with L2-normalised output (cosine via IP)."""
from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class QwenEmbedder:
    """Lazy-loaded sentence-transformers wrapper.

    Default model: Qwen/Qwen3-Embedding-0.6B (first run downloads ~1.2GB to ~/.cache/huggingface).
    Override via env EMBEDDING_MODEL.
    """

    _model = None
    _model_id = ""

    def __init__(self, model_id: str | None = None) -> None:
        self._target_model_id = model_id or os.environ.get(
            "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
        )

    def _load(self) -> None:
        if QwenEmbedder._model is not None and QwenEmbedder._model_id == self._target_model_id:
            return
        from sentence_transformers import SentenceTransformer  # heavy import deferred
        logger.info("loading embedding model %s ...", self._target_model_id)
        QwenEmbedder._model = SentenceTransformer(self._target_model_id)
        QwenEmbedder._model_id = self._target_model_id
        logger.info("embedding dim = %d", QwenEmbedder._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        """Returns L2-normalised float32 vectors of shape (N, D)."""
        self._load()
        assert QwenEmbedder._model is not None
        vecs = QwenEmbedder._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        return vecs

    @property
    def model_id(self) -> str:
        return self._target_model_id
