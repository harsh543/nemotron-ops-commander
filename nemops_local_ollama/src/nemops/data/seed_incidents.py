"""Seed ChromaDB with GPU incident patterns for RAG search."""

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chromadb")
COLLECTION_NAME = "gpu_incidents"


def main():
    """Load incidents from JSON and embed into ChromaDB."""
    incidents_file = Path(__file__).parent / "incidents.json"

    with open(incidents_file) as f:
        incidents = json.load(f)

    # Create ChromaDB client with sentence-transformer embeddings
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete existing collection if it exists
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except (ValueError, chromadb.errors.NotFoundError):
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Add incidents — embed the full text (symptoms + root cause + remediation)
    documents = []
    metadatas = []
    ids = []

    for inc in incidents:
        # Create a rich document for embedding
        doc = (
            f"Title: {inc['title']}\n"
            f"Severity: {inc['severity']}\n"
            f"Symptoms: {inc['symptoms']}\n"
            f"Root Cause: {inc['root_cause']}\n"
            f"Remediation: {inc['remediation']}"
        )
        documents.append(doc)
        metadatas.append(
            {
                "title": inc["title"],
                "severity": inc["severity"],
                "symptoms": inc["symptoms"],
                "root_cause": inc["root_cause"],
                "remediation": inc["remediation"],
                "xid_codes": inc.get("xid_codes", ""),
                "predictive_window": inc.get("predictive_window", ""),
            }
        )
        ids.append(inc["id"])

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅ Seeded {len(incidents)} GPU incidents into ChromaDB at {CHROMA_PATH}")


if __name__ == "__main__":
    main()
