"""Unit tests for RAG components."""

from __future__ import annotations

from rag.retriever import RAGRetriever


class DummyEmbedding:
    def embed(self, text: str):
        return [0.1, 0.2, 0.3]


class DummyStore:
    def search(self, query_vector, top_k=5):
        class Hit:
            def __init__(self):
                self.id = "1"
                self.score = 0.9
                self.payload = {"summary": "test"}

        return [Hit()]


def test_retriever_query():
    retriever = RAGRetriever(DummyEmbedding(), DummyStore())
    response = retriever.query("test")
    assert response.results[0].payload["summary"] == "test"
