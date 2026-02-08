"""Telemetry middleware for metrics and tracing."""

from __future__ import annotations

import time

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from observability.metrics import record_request

logger = structlog.get_logger(__name__)


class TelemetryMiddleware(BaseHTTPMiddleware):
    """Collect metrics for each request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.time()
        response = await call_next(request)
        latency_ms = (time.time() - start) * 1000

        record_request(request.url.path, str(response.status_code), latency_ms)
        logger.info(
            "request.complete",
            path=request.url.path,
            status=response.status_code,
            latency_ms=latency_ms,
        )
        return response
