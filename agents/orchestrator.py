"""LangGraph orchestrator for multi-agent workflows."""

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from agents.incident_triager import IncidentTriagerAgent
from agents.log_analyzer import LogAnalyzerAgent
from agents.remediation_suggester import RemediationSuggesterAgent
from models.nemotron_client import NemotronClient, NemotronConfig


class Orchestrator:
    """Multi-agent orchestration pipeline."""

    def __init__(self, client: NemotronClient) -> None:
        self.client = client
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(dict)
        graph.add_node("analyze", self._analyze)
        graph.add_node("triage", self._triage)
        graph.add_node("remediate", self._remediate)
        graph.set_entry_point("analyze")
        graph.add_edge("analyze", "triage")
        graph.add_edge("triage", "remediate")
        graph.add_edge("remediate", END)
        return graph.compile()

    async def _analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        agent = LogAnalyzerAgent(self.client)
        result = await agent.run(state["analysis_payload"])
        state["analysis_result"] = result.model_dump()
        return state

    async def _triage(self, state: Dict[str, Any]) -> Dict[str, Any]:
        agent = IncidentTriagerAgent(self.client)
        result = await agent.run(state["triage_payload"])
        state["triage_result"] = result.model_dump()
        return state

    async def _remediate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        agent = RemediationSuggesterAgent(self.client)
        result = await agent.run(state["remediation_payload"])
        state["remediation_result"] = result.model_dump()
        return state

    async def run(self, payloads: Dict[str, Any]) -> Dict[str, Any]:
        return await self.graph.ainvoke(payloads)


def build_default_orchestrator() -> Orchestrator:
    client = NemotronClient(NemotronConfig())
    return Orchestrator(client)
