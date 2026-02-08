"""RAG retrieval logic using ChromaDB and embeddings."""

from __future__ import annotations

from typing import List

import structlog

from models.schemas import RAGQueryResponse, RAGQueryResult
from rag.embeddings import EmbeddingService
from rag.vector_store import ChromaVectorStore

logger = structlog.get_logger(__name__)


class RAGRetriever:
    """Retrieves relevant incidents using semantic search over ChromaDB."""

    def __init__(self, embedding: EmbeddingService, store: ChromaVectorStore) -> None:
        self.embedding = embedding
        self.store = store

    def query(self, text: str, top_k: int = 5) -> RAGQueryResponse:
        """Retrieve similar incidents for a query string."""

        vector = self.embedding.embed(text)
        results = self.store.search(query_embedding=vector, top_k=top_k)

        formatted: List[RAGQueryResult] = []
        for hit in results:
            formatted.append(
                RAGQueryResult(
                    id=hit["id"],
                    score=hit["score"],
                    payload=hit.get("metadata", {}),
                )
            )

        logger.info("rag.query", top_k=top_k, result_count=len(formatted))
        return RAGQueryResponse(results=formatted)

    def query_text(self, text: str, top_k: int = 5) -> str:
        """Retrieve similar incidents and return as formatted text for LLM context."""

        vector = self.embedding.embed(text)
        results = self.store.search(query_embedding=vector, top_k=top_k)

        if not results:
            return "No similar historical incidents found."

        lines = ["Similar historical incidents:"]
        for i, hit in enumerate(results, 1):
            meta = hit.get("metadata", {})
            lines.append(
                f"\n{i}. [{meta.get('source', 'unknown')}] {meta.get('title', 'N/A')} "
                f"(score: {hit['score']:.2f})\n"
                f"   Resolution: {meta.get('resolution', 'N/A')}"
            )

        return "\n".join(lines)
