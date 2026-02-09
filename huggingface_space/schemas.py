"""Pydantic models for structured agent I/O."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    timestamp: str
    source: str
    message: str
    labels: Dict[str, str] = Field(default_factory=dict)


class Finding(BaseModel):
    severity: str
    summary: str
    evidence: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class LogAnalysisResult(BaseModel):
    findings: List[Finding] = Field(default_factory=list)
    root_cause: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)
    raw_text: str = ""
    latency_ms: float = 0.0


class TriageResult(BaseModel):
    priority: str = "P3"
    impact: str = "unknown"
    likely_services: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    raw_text: str = ""
    latency_ms: float = 0.0


class RemediationSuggestion(BaseModel):
    action: str
    rationale: str
    risk: str


class OptimizationResult(BaseModel):
    bottleneck: str = "unknown"
    severity: str = "medium"
    recommendations: List[RemediationSuggestion] = Field(default_factory=list)
    raw_text: str = ""
    latency_ms: float = 0.0


class RAGResult(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any] = Field(default_factory=dict)
