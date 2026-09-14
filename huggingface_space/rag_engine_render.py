"""
rag_engine_render.py -- Render deployment's RAG engine.

Same interface as rag_engine.py, but embeddings come from Nebius Token
Factory instead of a local sentence-transformers model. The free Render
tier's 512MB RAM can't hold torch + a 1B-param embedding model (measured:
OOM at startup); this routes embeddings through the same hosted Nebius
API already used for the LLM, so nothing heavy loads into this process's
memory. The HF Space keeps its own local ZeroGPU embeddings unchanged --
this file is only imported when EMBEDDING_BACKEND=nebius (set in
render.yaml).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

from schemas import RAGResult

DATA_DIR = Path(__file__).parent / "data" / "sample_incidents"
CHROMA_DIR = "/tmp/chroma_nemops_render"
COLLECTION_NAME = "incidents"
NEBIUS_EMBEDDING_MODEL = os.environ.get("NEBIUS_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
EMBEDDING_MODEL = NEBIUS_EMBEDDING_MODEL  # shown in the app's header markdown


def _embed(texts: List[str]) -> List[List[float]]:
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
        api_key=os.environ["NEBIUS_API_KEY"],
    )
    resp = client.embeddings.create(model=NEBIUS_EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


class RAGEngine:
    """Embedded ChromaDB + Nebius-hosted embeddings (no local model)."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._indexed = False

    def index_incidents(self, force: bool = False) -> int:
        if not force and self.collection.count() > 0:
            self._indexed = True
            return self.collection.count()

        incidents = self._load_incidents()
        if not incidents:
            return 0

        ids, documents, metadatas = [], [], []
        for inc in incidents:
            doc_text = f"{inc.get('summary', '')} {inc.get('root_cause', '')} {inc.get('resolution', '')}"
            ids.append(str(inc["id"]))
            documents.append(doc_text)
            metadatas.append(
                {
                    "title": inc.get("summary", inc.get("title", "")),
                    "source": inc.get("source", "internal"),
                    "severity": inc.get("severity", "unknown"),
                    "resolution": inc.get("resolution", ""),
                    "service": inc.get("service", ""),
                    "tags": ",".join(inc.get("tags", [])),
                }
            )

        embeddings = _embed(documents)
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        self._indexed = True
        return len(incidents)

    def search(self, query: str, top_k: int = 5) -> List[RAGResult]:
        if not self._indexed:
            self.index_incidents()

        vector = _embed([query])[0]
        results = self.collection.query(
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
            hits.append(RAGResult(id=doc_id, score=round(score, 4), payload=metadata))

        return hits

    def search_text(self, query: str, top_k: int = 3) -> str:
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


_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
