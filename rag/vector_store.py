"""ChromaDB vector store integration (embedded, no server needed)."""

from __future__ import annotations

from typing import List, Optional

import structlog

logger = structlog.get_logger(__name__)


class ChromaVectorStore:
    """Wrapper around ChromaDB for incident embeddings.

    Uses ChromaDB in embedded mode — no external server required.
    """

    def __init__(
        self,
        collection_name: str = "incidents",
        persist_directory: str = "./chroma_storage",
    ) -> None:
        import chromadb

        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "chroma.init",
            collection=collection_name,
            persist=persist_directory,
            count=self.collection.count(),
        )

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        """Upsert vectors into ChromaDB."""

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("chroma.upsert", count=len(ids))

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[dict]:
        """Search for similar vectors and return results with scores."""

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[dict] = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        for i, doc_id in enumerate(results["ids"][0]):
            score = 1.0 - (results["distances"][0][i] if results["distances"] else 0)
            hits.append(
                {
                    "id": doc_id,
                    "score": round(score, 4),
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                }
            )

        return hits

    def count(self) -> int:
        """Return total documents in collection."""
        return self.collection.count()

    def delete_collection(self) -> None:
        """Delete the entire collection (for re-indexing)."""
        self.client.delete_collection(self.collection.name)
        logger.info("chroma.delete_collection")

    def health_check(self) -> bool:
        """Check if ChromaDB is accessible."""
        try:
            self.collection.count()
            return True
        except Exception:
            return False
