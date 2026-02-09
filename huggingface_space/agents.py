"""
Multi-agent SRE system adapted for HuggingFace Spaces.

4 specialized agents that use the HF Inference API instead of local SGLang/Nemotron.
Each agent: builds a prompt -> calls LLM -> parses structured JSON -> returns result.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from inference import get_client
from schemas import (
    Finding,
    LogAnalysisResult,
    OptimizationResult,
    RemediationSuggestion,
    TriageResult,
)


def _repair_json_candidate(candidate: str) -> str:
    """Best-effort cleanup for malformed JSON from LLM outputs."""
    # Remove any trailing text after the last JSON brace
    if "{" in candidate and "}" in candidate:
        candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]

    cleaned_lines = []
    for line in candidate.splitlines():
        stripped = line.strip()
        # Replace invalid expressions like: "calculated": (max(0.9, 1)
        if ":" in stripped and "(" in stripped and stripped.count('"') >= 2:
            key = line.split(":", 1)[0]
            cleaned_lines.append(f"{key}: null,")
            continue
        cleaned_lines.append(line)

    candidate = "\n".join(cleaned_lines)
    # Remove trailing commas before closing braces/brackets
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    # Balance extra closing braces/brackets if the model adds one
    while candidate.count("}") > candidate.count("{"):
        candidate = candidate.rsplit("}", 1)[0] + "}"
    while candidate.count("]") > candidate.count("["):
        candidate = candidate.rsplit("]", 1)[0] + "]"
    return candidate


def _extract_findings_from_text(text: str) -> list[dict[str, Any]]:
    """Extract findings array from partially malformed JSON text."""
    match = re.search(r"\"findings\"\s*:\s*(\[[\s\S]*?\])", text)
    if not match:
        return []

    candidate = "{" + f"\"findings\": {match.group(1)}" + "}"
    candidate = _repair_json_candidate(candidate)
    try:
        data = json.loads(candidate)
        findings = data.get("findings", [])
        return findings if isinstance(findings, list) else []
    except json.JSONDecodeError:
        return []


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM output, handling markdown and minor malformations."""

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        candidate = _repair_json_candidate(match.group(1))
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = _repair_json_candidate(match.group(0))
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {}


# ---------------------------------------------------------------------------
# Agent 1: Log Analyzer
# ---------------------------------------------------------------------------

LOG_ANALYZER_SYSTEM = (
    "You are a senior SRE assistant. Analyze the provided logs and return ONLY valid JSON "
    "with this exact schema: {\"findings\": [{\"severity\": \"critical|high|medium|low\", "
    "\"summary\": \"...\", \"evidence\": [\"log line...\"], \"confidence\": 0.0-1.0}], "
    "\"root_cause\": \"...\", \"recommendations\": [\"...\"]}. "
    "Be concise. Cite evidence from actual log lines. Do not wrap in markdown."
)


def analyze_logs(
    log_text: str, system: str = "unknown", environment: str = "production"
) -> LogAnalysisResult:
    """Analyze log entries and return structured findings."""

    # Parse log lines
    log_lines = [line.strip() for line in log_text.strip().splitlines() if line.strip()]
    if not log_lines:
        return LogAnalysisResult(raw_text="No logs provided.")

    prompt = (
        f"Analyze the following logs from service '{system}' in {environment} environment.\n"
        f"Return JSON with findings, root_cause, and recommendations.\n\n"
        f"Logs:\n" + "\n".join(log_lines)
    )

    client = get_client()
    try:
        text, latency_ms = client.generate(prompt, system_prompt=LOG_ANALYZER_SYSTEM)
        data = _extract_json(text)
        if not data or not data.get("findings"):
            data = {"findings": _extract_findings_from_text(text)}

        findings = []
        for f in data.get("findings", []):
            findings.append(Finding(
                severity=f.get("severity", "medium"),
                summary=f.get("summary", ""),
                evidence=f.get("evidence", []),
                confidence=float(f.get("confidence", 0.5)),
            ))

        return LogAnalysisResult(
            findings=findings,
            root_cause=data.get("root_cause"),
            recommendations=data.get("recommendations", []),
            raw_text=text,
            latency_ms=latency_ms,
        )
    except Exception as e:
        return LogAnalysisResult(
            raw_text=f"Inference error: {e}",
            root_cause="Could not analyze — inference API unavailable.",
            recommendations=["Retry in a few seconds", "Check HF_TOKEN is set"],
        )


# ---------------------------------------------------------------------------
# Agent 2: Incident Triager
# ---------------------------------------------------------------------------

TRIAGER_SYSTEM = (
    "You are an incident commander for a production SRE team. "
    "Return ONLY valid JSON with this schema: "
    "{\"priority\": \"P0|P1|P2|P3|P4\", \"impact\": \"...\", "
    "\"likely_services\": [\"...\"], \"next_steps\": [\"...\"]}. "
    "Be concise and actionable. Do not wrap in markdown."
)


def triage_incident(
    title: str,
    description: str,
    error_rate: float = 0.0,
    latency_p99: float = 0.0,
) -> TriageResult:
    """Triage an incident and return priority classification."""

    prompt = (
        f"Triage this incident and return JSON with priority, impact, likely_services, next_steps.\n\n"
        f"Title: {title}\n"
        f"Description: {description}\n"
        f"Metrics: error_rate={error_rate:.2%}, p99_latency={latency_p99:.0f}ms\n"
    )

    client = get_client()
    try:
        text, latency_ms = client.generate(prompt, system_prompt=TRIAGER_SYSTEM)
        data = _extract_json(text)

        return TriageResult(
            priority=data.get("priority", "P3"),
            impact=data.get("impact", "unknown"),
            likely_services=data.get("likely_services", []),
            next_steps=data.get("next_steps", []),
            raw_text=text,
            latency_ms=latency_ms,
        )
    except Exception as e:
        return TriageResult(
            raw_text=f"Inference error: {e}",
            impact="Could not triage — inference API unavailable.",
            next_steps=["Retry in a few seconds", "Check HF_TOKEN is set"],
        )


# ---------------------------------------------------------------------------
# Agent 3: Performance Optimizer
# ---------------------------------------------------------------------------

OPTIMIZER_SYSTEM = (
    "You are a senior performance engineer. Analyze system metrics and return ONLY valid JSON: "
    "{\"bottleneck\": \"...\", \"severity\": \"critical|high|medium|low\", "
    "\"recommendations\": [{\"action\": \"...\", \"rationale\": \"...\", \"risk\": \"low|medium|high\"}]}. "
    "Focus on actionable, specific recommendations. Do not wrap in markdown."
)


def _extract_recommendations_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract recommendations array from partially malformed JSON text."""
    match = re.search(r"\"recommendations\"\s*:\s*(\[[\s\S]*?\])", text)
    if not match:
        return []
    candidate = "{" + f'"recommendations": {match.group(1)}' + "}"
    candidate = _repair_json_candidate(candidate)
    try:
        data = json.loads(candidate)
        recs = data.get("recommendations", [])
        return recs if isinstance(recs, list) else []
    except json.JSONDecodeError:
        return []


def _extract_field_from_text(text: str, field: str) -> str:
    """Extract a single string field value from partially malformed JSON."""
    match = re.search(rf'"{field}"\s*:\s*"([^"]*)"', text)
    return match.group(1) if match else ""


def optimize_performance(
    cpu: float,
    memory: float,
    gpu: float,
    service: str = "unknown",
    context: str = "",
) -> OptimizationResult:
    """Analyze system metrics and return optimization recommendations."""

    metrics_str = f"CPU: {cpu:.0f}%, Memory: {memory:.0f}%, GPU: {gpu:.0f}%"
    prompt = (
        f"Analyze metrics for service '{service}' and provide optimization recommendations.\n\n"
        f"Metrics: {metrics_str}\n"
        f"Context: {context or 'production workload'}\n\n"
        f"Return JSON with bottleneck, severity, and recommendations (each with action, rationale, risk)."
    )

    client = get_client()
    try:
        text, latency_ms = client.generate(prompt, system_prompt=OPTIMIZER_SYSTEM)
        data = _extract_json(text)

        # Fallback: if full JSON parsing failed, extract fields individually
        if not data or (not data.get("bottleneck") and not data.get("recommendations")):
            recs_raw = _extract_recommendations_from_text(text)
            data = {
                "bottleneck": data.get("bottleneck") or _extract_field_from_text(text, "bottleneck") or "see analysis below",
                "severity": data.get("severity") or _extract_field_from_text(text, "severity") or "medium",
                "recommendations": recs_raw or data.get("recommendations", []),
            }

        recommendations = []
        for r in data.get("recommendations", []):
            if isinstance(r, dict):
                recommendations.append(RemediationSuggestion(
                    action=r.get("action", ""),
                    rationale=r.get("rationale", ""),
                    risk=r.get("risk", "medium"),
                ))
            elif isinstance(r, str):
                recommendations.append(RemediationSuggestion(
                    action=r, rationale="", risk="medium",
                ))

        return OptimizationResult(
            bottleneck=data.get("bottleneck", "unknown"),
            severity=data.get("severity", "medium"),
            recommendations=recommendations,
            raw_text=text,
            latency_ms=latency_ms,
        )
    except Exception as e:
        return OptimizationResult(
            raw_text=f"Inference error: {e}",
            bottleneck="Could not analyze — inference API unavailable.",
            recommendations=[],
        )
