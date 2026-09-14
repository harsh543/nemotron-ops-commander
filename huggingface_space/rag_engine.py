"""
RAG pipeline: ChromaDB vector store + sentence-transformers embeddings.

Indexes 30 real-world SRE incidents and provides semantic search.
Fully embedded — no external database or server needed.
"""

from __future__ import annotations

import spaces
import json
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from schemas import RAGResult

DATA_DIR = Path(__file__).parent / "data" / "sample_incidents"
CHROMA_DIR = "/tmp/chroma_nemops_nvidia"  # New dir for NVIDIA embeddings
COLLECTION_NAME = "incidents"
# NVIDIA llama-nemotron-embed-1b-v2: +20-36% better retrieval, 16× longer context (8K tokens)
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
EMBEDDING_DIM = 1024  # Matryoshka: configurable 384-2048, using 1024 for balance


class RAGEngine:
    """Embedded RAG engine using ChromaDB + NVIDIA sentence-transformers."""

    def __init__(self):
        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
            model_kwargs={"dtype": torch.bfloat16},
            device="cuda",
        )
        self.client = None
        self.collection = None
        self._indexed = False

    def _get_collection(self):
        """Open Chroma lazily inside the ZeroGPU worker process."""
        if self.collection is None:
            self.client = chromadb.PersistentClient(path=CHROMA_DIR)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self.collection

    def index_incidents(self, force: bool = False) -> int:
        """Index all incident JSON files into ChromaDB. Returns count."""

        collection = self._get_collection()

        if not force and collection.count() > 0:
            self._indexed = True
            return collection.count()

        incidents = self._load_incidents()
        if not incidents:
            return 0

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for inc in incidents:
            doc_text = (
                f"{inc.get('summary', '')} "
                f"{inc.get('root_cause', '')} "
                f"{inc.get('resolution', '')}"
            )
            ids.append(str(inc["id"]))
            documents.append(doc_text)
            # Encode with configured dimension (Matryoshka embeddings)
            embedding = self.embedder.encode_document(
                doc_text, convert_to_tensor=False
            )
            # For llama-nemotron-embed-1b-v2, embeddings are already at target dim
            embeddings.append(embedding.tolist())
            metadatas.append({
                "title": inc.get("summary", inc.get("title", "")),
                "source": inc.get("source", "internal"),
                "severity": inc.get("severity", "unknown"),
                "resolution": inc.get("resolution", ""),
                "service": inc.get("service", ""),
                "tags": ",".join(inc.get("tags", [])),
            })

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        self._indexed = True
        return len(incidents)

    def search(self, query: str, top_k: int = 5) -> List[RAGResult]:
        """Semantic search over indexed incidents."""

        if not self._indexed:
            self.index_incidents()

        # Encode query with same dimension
        vector = self.embedder.encode_query(query, convert_to_tensor=False).tolist()
        results = self._get_collection().query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0
            score = 1.0 - distance
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            hits.append(RAGResult(
                id=doc_id,
                score=round(score, 4),
                payload=metadata,
            ))

        return hits

    def search_text(self, query: str, top_k: int = 3) -> str:
        """Search and return formatted text for LLM context injection."""

        hits = self.search(query, top_k=top_k)
        if not hits:
            return "No similar historical incidents found."

        lines = ["Similar historical incidents:"]
        for i, hit in enumerate(hits, 1):
            p = hit.payload
            lines.append(
                f"\n{i}. [{p.get('source', '?')}] {p.get('title', hit.id)} "
                f"(severity: {p.get('severity', '?')}, score: {hit.score:.2f})\n"
                f"   Resolution: {p.get('resolution', 'N/A')}"
            )
        return "\n".join(lines)

    def count(self) -> int:
        """Return the bundled corpus size without running GPU inference."""
        return len(list(DATA_DIR.glob("*.json")))

    def _load_incidents(self) -> List[Dict]:
        incidents = []
        if not DATA_DIR.exists():
            return incidents
        for path in sorted(DATA_DIR.glob("*.json")):
            try:
                incidents.append(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return incidents


# Singleton
_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
