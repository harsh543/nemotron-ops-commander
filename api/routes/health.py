"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import generate_latest

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic liveness check."""

    return {"status": "ok"}


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""

    return Response(content=generate_latest(), media_type="text/plain")
