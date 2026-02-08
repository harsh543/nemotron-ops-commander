"""Unit tests for agents."""

from __future__ import annotations

import pytest

from agents.log_analyzer import LogAnalyzerAgent
from agents.incident_triager import IncidentTriagerAgent
from models.nemotron_client import NemotronResponse


class DummyClient:
    async def generate(self, prompt: str, system_prompt=None, structured_output=None):
        return NemotronResponse(text="{}", metadata={}, latency_ms=1.0)


@pytest.mark.asyncio
async def test_log_analyzer_runs():
    agent = LogAnalyzerAgent(DummyClient())
    payload = {"logs": [{"timestamp": "t", "source": "s", "message": "m"}]}
    result = await agent.run(payload)
    assert result.raw_text is not None


@pytest.mark.asyncio
async def test_triager_runs():
    agent = IncidentTriagerAgent(DummyClient())
    payload = {
        "incident_id": "INC-1",
        "title": "t",
        "description": "d",
        "logs": [],
        "metrics": {},
    }
    result = await agent.run(payload)
    assert result.raw_text is not None
