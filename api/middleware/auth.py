"""Simple API key authentication middleware."""

from __future__ import annotations

import time

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config.settings import get_settings

logger = structlog.get_logger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Enforces API key auth and basic rate limiting."""

    _bucket = {}

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        api_key = request.headers.get("X-API-Key")
        if settings.api_key and api_key != settings.api_key:
            logger.warning("auth.failed", path=request.url.path)
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)

        key = request.client.host if request.client else "unknown"
        now = time.time()
        window = 60
        bucket = self._bucket.setdefault(key, [])
        bucket[:] = [ts for ts in bucket if now - ts < window]
        if len(bucket) >= settings.rate_limit_per_minute:
            logger.warning("rate_limit.exceeded", path=request.url.path)
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        bucket.append(now)

        return await call_next(request)
