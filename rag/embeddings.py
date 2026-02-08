"""Embedding service for incident retrieval."""

from __future__ import annotations

from typing import List

import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """Sentence-transformers embedding wrapper."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""

        vector = self.model.encode([text])[0]
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)
