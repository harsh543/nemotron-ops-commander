"""Index historical incidents into ChromaDB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import structlog

from rag.embeddings import EmbeddingService
from rag.vector_store import ChromaVectorStore

logger = structlog.get_logger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_incidents"


class IncidentIndexer:
    """Indexes incident JSON files into ChromaDB."""

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        persist_directory: str = "./chroma_storage",
    ) -> None:
        self.data_dir = data_dir
        self.embedding = EmbeddingService()
        self.store = ChromaVectorStore(persist_directory=persist_directory)

    def _load_incidents(self) -> List[dict]:
        incidents: List[dict] = []
        for path in sorted(self.data_dir.glob("*.json")):
            try:
                incidents.append(json.loads(path.read_text()))
            except json.JSONDecodeError:
                logger.warning("indexer.skip_invalid", path=str(path))
        return incidents

    def run(self, reset: bool = False) -> int:
        """Index all incidents. Returns count of indexed documents.

        Args:
            reset: If True, delete existing collection before indexing.
        """
        if reset:
            try:
                self.store.delete_collection()
                self.store = ChromaVectorStore(
                    persist_directory=self.store.persist_directory
                )
            except Exception:
                pass

        incidents = self._load_incidents()
        if not incidents:
            logger.warning("indexer.no_incidents", data_dir=str(self.data_dir))
            return 0

        ids: List[str] = []
        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[dict] = []

        for inc in incidents:
            doc_text = f"{inc.get('title', '')} {inc.get('summary', '')} {inc.get('resolution', '')}"
            ids.append(str(inc["id"]))
            documents.append(doc_text)
            embeddings.append(self.embedding.embed(doc_text))
            metadatas.append(
                {
                    "title": inc.get("title", ""),
                    "source": inc.get("source", "unknown"),
                    "severity": inc.get("severity", "unknown"),
                    "resolution": inc.get("resolution", ""),
                    "services": ",".join(inc.get("services", [])),
                    "tags": ",".join(inc.get("tags", [])),
                }
            )

        self.store.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("indexer.complete", count=len(incidents))
        return len(incidents)
