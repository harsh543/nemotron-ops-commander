"""Remediation suggestion agent."""

from __future__ import annotations

from typing import Any, Dict

import structlog
from pydantic import BaseModel

from agents.base_agent import BaseAgent
from models.schemas import RemediationResponse, RemediationSuggestion

logger = structlog.get_logger(__name__)


class RemediationOutput(BaseModel):
    """Structured output schema for remediation."""

    suggestions: list[RemediationSuggestion]


class RemediationSuggesterAgent(BaseAgent):
    """Agent that proposes remediation steps."""

    name = "remediation-suggester"
    system_prompt = (
        "You are a senior SRE. Return only JSON with a list of remediation "
        "suggestions with action, rationale, and risk."
    )

    async def run(self, payload: Dict[str, Any]) -> BaseModel:
        """Run remediation suggestions."""

        prompt = self._build_prompt(payload)
        text, latency_ms = await self._generate_text(prompt)

        try:
            parsed = self._parse_json(text, RemediationOutput)
        except Exception:  # noqa: BLE001
            logger.warning("remediation.parse_fallback", raw=text)
            parsed = RemediationOutput(suggestions=[])

        return RemediationResponse(
            suggestions=parsed.suggestions,
            raw_text=text,
            latency_ms=latency_ms,
        )

    def _build_prompt(self, payload: Dict[str, Any]) -> str:
        return (
            "Provide remediation suggestions. Return JSON with key 'suggestions'. "
            "Each suggestion needs action, rationale, risk.\n\n"
            f"Context: {payload}"
        )
