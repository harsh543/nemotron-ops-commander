"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware.auth import AuthMiddleware
from api.middleware.telemetry import TelemetryMiddleware
from api.routes.analyze import router as analyze_router
from api.routes.health import router as health_router
from api.routes.optimize import router as optimize_router
from api.routes.rag import router as rag_router
from api.routes.triage import router as triage_router
from config.settings import get_settings
from observability.logging import configure_logging
from observability.tracing import configure_tracing, instrument_fastapi

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan hooks."""

    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing(settings)
    logger.info("app.startup", env=settings.app_env)
    yield
    logger.info("app.shutdown")


def create_app() -> FastAPI:
    """Create FastAPI application."""

    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="AI-Powered Incident Response System using NVIDIA Nemotron",
        lifespan=lifespan,
    )

    # Middleware (order matters: last added = first executed)
    app.add_middleware(TelemetryMiddleware)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    instrument_fastapi(app)

    # Routes
    app.include_router(health_router, prefix="/health", tags=["health"])
    app.include_router(analyze_router, prefix="/analyze", tags=["analysis"])
    app.include_router(triage_router, prefix="/triage", tags=["triage"])
    app.include_router(optimize_router, prefix="/optimize", tags=["optimize"])
    app.include_router(rag_router, prefix="/rag", tags=["rag"])

    return app


app = create_app()


def run() -> None:
    """Run the server via CLI."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "development",
    )


if __name__ == "__main__":
    run()
