"""Tests for NemOps agent module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from nemops.agent import (
    TOOL_DEFINITIONS,
    TOOLS,
    execute_tool,
    load_system_prompt,
)


class TestToolRegistry:
    """Verify tool registry is consistent."""

    def test_all_tools_registered(self):
        assert "gpu_health_check" in TOOLS
        assert "search_incidents" in TOOLS
        assert "run_diagnostic" in TOOLS
        assert "generate_alert" in TOOLS

    def test_tool_definitions_match_registry(self):
        defined_names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        registered_names = set(TOOLS.keys())
        assert defined_names == registered_names

    def test_tool_definitions_have_required_fields(self):
        for td in TOOL_DEFINITIONS:
            assert td["type"] == "function"
            func = td["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func


class TestExecuteTool:
    """Test execute_tool wrapper."""

    def test_execute_known_tool(self):
        result = json.loads(execute_tool("gpu_health_check", {}))
        assert "gpus" in result or "error" in result

    def test_execute_unknown_tool(self):
        result = json.loads(execute_tool("nonexistent_tool", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_execute_tool_with_bad_args(self):
        result = json.loads(execute_tool("run_diagnostic", {"test_name": "fake_test"}))
        # Should return error about unknown test
        assert "error" in result or "available_tests" in result


class TestSystemPrompt:
    """Verify system prompt content."""

    def test_prompt_mentions_key_concepts(self):
        prompt = load_system_prompt()
        assert "NemOps" in prompt
        assert "GPU" in prompt
        assert "gpu_health_check" in prompt
        assert "search_incidents" in prompt
        assert "run_diagnostic" in prompt
        assert "generate_alert" in prompt

    def test_prompt_is_nonempty(self):
        prompt = load_system_prompt()
        assert len(prompt) > 100


class TestRunAgentOffline:
    """Test agent behavior when Ollama is offline."""

    @patch("nemops.agent.NemotronClient")
    def test_agent_returns_error_when_ollama_down(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.health_check.return_value = False
        mock_client.model = "nemotron-3-nano:30b-cloud"
        mock_client_cls.return_value = mock_client

        from nemops.agent import run_agent

        result = run_agent("test query", verbose=False)
        assert "error" in result
        assert result["tool_calls"] == []
