"""Log analysis agent implementation."""

from __future__ import annotations

import json
from typing import Any, Dict

import structlog
from pydantic import BaseModel

from agents.base_agent import BaseAgent
from models.schemas import Finding, LogAnalysisRequest, LogAnalysisResponse

logger = structlog.get_logger(__name__)


class LogAnalyzerOutput(BaseModel):
    """Structured output schema for the log analyzer."""

    findings: list[Finding]
    root_cause: str | None = None
    recommendations: list[str]


class LogAnalyzerAgent(BaseAgent):
    """Agent that analyzes logs and returns structured findings."""

    name = "log-analyzer"
    system_prompt = (
        "You are a senior SRE assistant. Analyze logs and return only JSON matching the schema."
        " Be concise, cite evidence lines, and avoid speculation."
    )

    async def run(self, payload: Dict[str, Any]) -> BaseModel:
        """Run log analysis and return structured results."""

        request = LogAnalysisRequest.model_validate(payload)
        prompt = self._build_prompt(request)
        text, latency_ms = await self._generate_text(prompt)

        try:
            parsed = self._parse_json(text, LogAnalyzerOutput)
        except Exception:  # noqa: BLE001
            logger.warning("log_analyzer.parse_fallback", raw=text)
            parsed = LogAnalyzerOutput(findings=[], root_cause=None, recommendations=[])

        return LogAnalysisResponse(
            findings=parsed.findings,
            root_cause=parsed.root_cause,
            recommendations=parsed.recommendations,
            raw_text=text,
            latency_ms=latency_ms,
        )

    def _build_prompt(self, request: LogAnalysisRequest) -> str:
        """Build prompt for log analysis."""

        log_lines = [
            f"[{entry.timestamp}] {entry.source} {entry.message}" for entry in request.logs
        ]
        return (
            "Analyze the following logs and return JSON with keys: "
            "findings (list), root_cause (string|null), recommendations (list). "
            "Each finding should include severity, summary, evidence, confidence.\n\n"
            f"System: {request.system or 'unknown'}\n"
            f"Environment: {request.environment or 'unknown'}\n\n"
            "Logs:\n"
            + "\n".join(log_lines)
        )
