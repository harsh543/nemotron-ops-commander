"""Pydantic models for API requests and responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    """Single log line with metadata."""

    timestamp: str
    source: str
    message: str
    labels: Dict[str, str] = Field(default_factory=dict)


class LogAnalysisRequest(BaseModel):
    """Request schema for log analysis."""

    logs: List[LogEntry]
    system: Optional[str] = None
    environment: Optional[str] = None


class Finding(BaseModel):
    """Structured finding from analysis."""

    severity: str
    summary: str
    evidence: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class LogAnalysisResponse(BaseModel):
    """Response schema for log analysis."""

    findings: List[Finding]
    root_cause: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)
    raw_text: str
    latency_ms: float


class IncidentTriageRequest(BaseModel):
    """Request schema for incident triage."""

    incident_id: str
    title: str
    description: str
    logs: List[LogEntry] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class IncidentTriageResponse(BaseModel):
    """Response schema for incident triage."""

    priority: str
    impact: str
    likely_services: List[str]
    next_steps: List[str]
    raw_text: str
    latency_ms: float


class RemediationSuggestion(BaseModel):
    """Remediation suggestion returned by the system."""

    action: str
    rationale: str
    risk: str


class RemediationResponse(BaseModel):
    """Response schema for remediation suggestions."""

    suggestions: List[RemediationSuggestion]
    raw_text: str
    latency_ms: float


class RAGQueryRequest(BaseModel):
    """Request schema for RAG queries."""

    query: str
    top_k: int = 5


class RAGQueryResult(BaseModel):
    """Single RAG result item."""

    id: str
    score: float
    payload: Dict[str, Any]


class RAGQueryResponse(BaseModel):
    """Response schema for RAG queries."""

    results: List[RAGQueryResult]
