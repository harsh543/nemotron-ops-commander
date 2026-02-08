"""Performance optimization agent implementation."""

from __future__ import annotations

from typing import Any, Dict, List

import structlog
from pydantic import BaseModel

from agents.base_agent import BaseAgent
from models.schemas import RemediationSuggestion

logger = structlog.get_logger(__name__)


class OptimizationRequest(BaseModel):
    """Request for performance optimization analysis."""

    metrics: Dict[str, float]
    service: str
    context: str = ""


class OptimizationResponse(BaseModel):
    """Structured optimization recommendations."""

    bottleneck: str
    severity: str
    recommendations: List[RemediationSuggestion]
    raw_text: str
    latency_ms: float


class OptimizerAgent(BaseAgent):
    """Agent that analyzes system metrics and recommends performance optimizations."""

    name = "optimizer"
    system_prompt = (
        "You are a senior performance engineer. Analyze system metrics and return only JSON "
        "with keys: bottleneck (string), severity (critical|high|medium|low), "
        "recommendations (list of {action, rationale, risk}). "
        "Focus on actionable, specific recommendations."
    )

    async def run(self, payload: Dict[str, Any]) -> BaseModel:
        """Analyze metrics and return optimization recommendations."""

        request = OptimizationRequest.model_validate(payload)
        prompt = self._build_prompt(request)
        text, latency_ms = await self._generate_text(prompt)

        try:
            parsed = self._parse_json(text, _OptimizerOutput)
        except Exception:
            logger.warning("optimizer.parse_fallback", raw=text)
            parsed = _OptimizerOutput(
                bottleneck="unknown",
                severity="medium",
                recommendations=[],
            )

        return OptimizationResponse(
            bottleneck=parsed.bottleneck,
            severity=parsed.severity,
            recommendations=parsed.recommendations,
            raw_text=text,
            latency_ms=latency_ms,
        )

    def _build_prompt(self, request: OptimizationRequest) -> str:
        metrics_str = "\n".join(f"  {k}: {v}" for k, v in request.metrics.items())
        return (
            f"Analyze the following system metrics for service '{request.service}' "
            f"and provide optimization recommendations.\n\n"
            f"Metrics:\n{metrics_str}\n\n"
            f"Context: {request.context or 'production workload'}\n\n"
            "Return JSON with: bottleneck, severity, recommendations "
            "(each with action, rationale, risk)."
        )


class _OptimizerOutput(BaseModel):
    bottleneck: str
    severity: str
    recommendations: List[RemediationSuggestion]
