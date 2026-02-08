"""Log analysis API endpoint."""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import structlog
from fastapi import APIRouter, HTTPException

from agents.log_analyzer import LogAnalyzerAgent
from config.settings import get_settings
from models.nemotron_client import NemotronClient, NemotronConfig
from models.schemas import LogAnalysisRequest, LogAnalysisResponse

logger = structlog.get_logger(__name__)
router = APIRouter()


def _build_agent() -> LogAnalyzerAgent:
    settings = get_settings()
    client = NemotronClient(
        NemotronConfig(
            model_name=settings.nemotron_model_name,
            device=settings.nemotron_device,
            max_new_tokens=settings.nemotron_max_new_tokens,
            temperature=settings.nemotron_temperature,
            top_p=settings.nemotron_top_p,
            use_sglang=settings.nemotron_use_sglang,
        )
    )
    return LogAnalyzerAgent(client)


@router.post("/", response_model=LogAnalysisResponse)
async def analyze_logs(request: LogAnalysisRequest) -> LogAnalysisResponse:
    """Analyze logs and return findings."""

    agent = _build_agent()
    try:
        result = await asyncio.wait_for(
            agent.run(request.model_dump()),
            timeout=30,
        )
        return result  # type: ignore[return-value]
    except asyncio.TimeoutError as exc:
        logger.error("api.analyze.timeout")
        raise HTTPException(status_code=504, detail="Analysis timed out") from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("api.analyze.error", error=str(exc))
        raise HTTPException(status_code=500, detail="Analysis failed") from exc
