"""Performance optimization API endpoint."""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, HTTPException

from agents.optimizer import OptimizerAgent, OptimizationRequest, OptimizationResponse
from config.settings import get_settings
from models.nemotron_client import NemotronClient, NemotronConfig

logger = structlog.get_logger(__name__)
router = APIRouter()


def _build_agent() -> OptimizerAgent:
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
    return OptimizerAgent(client)


@router.post("/", response_model=OptimizationResponse)
async def optimize(request: OptimizationRequest) -> OptimizationResponse:
    """Analyze system metrics and return optimization recommendations."""

    agent = _build_agent()
    try:
        result = await asyncio.wait_for(
            agent.run(request.model_dump()),
            timeout=30,
        )
        return result  # type: ignore[return-value]
    except asyncio.TimeoutError as exc:
        logger.error("api.optimize.timeout")
        raise HTTPException(status_code=504, detail="Optimization timed out") from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("api.optimize.error", error=str(exc))
        raise HTTPException(status_code=500, detail="Optimization failed") from exc
