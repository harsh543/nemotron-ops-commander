"""Incident triage agent implementation."""

from __future__ import annotations

from typing import Any, Dict

import structlog
from pydantic import BaseModel

from agents.base_agent import BaseAgent
from models.schemas import IncidentTriageRequest, IncidentTriageResponse

logger = structlog.get_logger(__name__)


class IncidentTriageOutput(BaseModel):
    """Structured output schema for incident triage."""

    priority: str
    impact: str
    likely_services: list[str]
    next_steps: list[str]


class IncidentTriagerAgent(BaseAgent):
    """Agent that triages incidents."""

    name = "incident-triager"
    system_prompt = (
        "You are an incident commander. Return only JSON with priority, impact, "
        "likely_services, next_steps. Be concise and actionable."
    )

    async def run(self, payload: Dict[str, Any]) -> BaseModel:
        """Run incident triage and return structured results."""

        request = IncidentTriageRequest.model_validate(payload)
        prompt = self._build_prompt(request)
        text, latency_ms = await self._generate_text(prompt)

        try:
            parsed = self._parse_json(text, IncidentTriageOutput)
        except Exception:  # noqa: BLE001
            logger.warning("triager.parse_fallback", raw=text)
            parsed = IncidentTriageOutput(
                priority="P3",
                impact="unknown",
                likely_services=[],
                next_steps=["Collect more data"],
            )

        return IncidentTriageResponse(
            priority=parsed.priority,
            impact=parsed.impact,
            likely_services=parsed.likely_services,
            next_steps=parsed.next_steps,
            raw_text=text,
            latency_ms=latency_ms,
        )

    def _build_prompt(self, request: IncidentTriageRequest) -> str:
        log_lines = [
            f"[{entry.timestamp}] {entry.source} {entry.message}" for entry in request.logs
        ]
        return (
            "Triage the incident and return JSON with keys: "
            "priority, impact, likely_services, next_steps.\n\n"
            f"Incident ID: {request.incident_id}\n"
            f"Title: {request.title}\n"
            f"Description: {request.description}\n\n"
            "Logs:\n"
            + "\n".join(log_lines)
        )
