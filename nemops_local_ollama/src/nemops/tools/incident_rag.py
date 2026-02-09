"""RAG-powered GPU incident search using ChromaDB + sentence-transformers."""

import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chromadb")
COLLECTION_NAME = "gpu_incidents"


def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection with sentence-transformer embeddings."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def search_incidents(query: str, top_k: int = 3) -> dict[str, Any]:
    """Search historical GPU incidents using semantic similarity.

    Args:
        query: Description of symptoms or issue to search for.
        top_k: Number of results to return.

    Returns:
        Dict with matching incidents, each containing title, symptoms,
        root_cause, remediation, severity, and similarity score.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return {
            "results": [],
            "message": "Incident database is empty. Run: nemops-seed",
        }

    results = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))

    incidents = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        incidents.append(
            {
                "id": results["ids"][0][i],
                "title": metadata.get("title", ""),
                "severity": metadata.get("severity", ""),
                "symptoms": metadata.get("symptoms", ""),
                "root_cause": metadata.get("root_cause", ""),
                "remediation": metadata.get("remediation", ""),
                "xid_codes": metadata.get("xid_codes", ""),
                "predictive_window": metadata.get("predictive_window", ""),
                "similarity_score": round(
                    1 - (results["distances"][0][i] if results["distances"] else 0), 3
                ),
            }
        )

    return {
        "query": query,
        "results": incidents,
        "total_incidents_in_db": collection.count(),
    }
