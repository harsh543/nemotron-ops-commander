"""RAG query endpoint."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException

from models.schemas import RAGQueryRequest, RAGQueryResponse
from rag.embeddings import EmbeddingService
from rag.retriever import RAGRetriever
from rag.vector_store import ChromaVectorStore

logger = structlog.get_logger(__name__)
router = APIRouter()


def _build_retriever() -> RAGRetriever:
    embedding = EmbeddingService()
    store = ChromaVectorStore()
    return RAGRetriever(embedding, store)


@router.post("/", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest) -> RAGQueryResponse:
    """Query historical incidents using semantic search."""

    retriever = _build_retriever()
    try:
        return retriever.query(request.query, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001
        logger.error("api.rag.error", error=str(exc))
        raise HTTPException(status_code=500, detail="RAG query failed") from exc
