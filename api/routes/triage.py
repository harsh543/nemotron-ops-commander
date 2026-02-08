"""Incident triage endpoint."""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, HTTPException

from agents.incident_triager import IncidentTriagerAgent
from config.settings import get_settings
from models.nemotron_client import NemotronClient, NemotronConfig
from models.schemas import IncidentTriageRequest, IncidentTriageResponse

logger = structlog.get_logger(__name__)
router = APIRouter()


def _build_agent() -> IncidentTriagerAgent:
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
    return IncidentTriagerAgent(client)


@router.post("/", response_model=IncidentTriageResponse)
async def triage_incident(request: IncidentTriageRequest) -> IncidentTriageResponse:
    """Triage an incident and return priority and next steps."""

    agent = _build_agent()
    try:
        result = await asyncio.wait_for(
            agent.run(request.model_dump()),
            timeout=30,
        )
        return result  # type: ignore[return-value]
    except asyncio.TimeoutError as exc:
        logger.error("api.triage.timeout")
        raise HTTPException(status_code=504, detail="Triage timed out") from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("api.triage.error", error=str(exc))
        raise HTTPException(status_code=500, detail="Triage failed") from exc
