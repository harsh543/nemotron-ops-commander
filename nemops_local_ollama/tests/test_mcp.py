"""Tests for NemOps MCP server."""

import json

import pytest

from nemops.mcp_server import app


@pytest.mark.asyncio
async def test_list_tools_returns_all_four():
    """MCP server should expose all 4 tools."""
    # Get the handler for list_tools
    tools = await app._tool_list_handler()
    names = {t.name for t in tools}
    assert names == {"gpu_health_check", "search_incidents", "run_diagnostic", "generate_alert"}


@pytest.mark.asyncio
async def test_call_gpu_health_check():
    """MCP server should execute gpu_health_check."""
    result = await app._tool_call_handler("gpu_health_check", {})
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert "gpus" in data or "error" in data


@pytest.mark.asyncio
async def test_call_run_diagnostic():
    """MCP server should execute run_diagnostic."""
    result = await app._tool_call_handler("run_diagnostic", {"test_name": "memory_stress"})
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert "test" in data


@pytest.mark.asyncio
async def test_call_unknown_tool():
    """MCP server should handle unknown tools gracefully."""
    result = await app._tool_call_handler("fake_tool", {})
    assert "Unknown tool" in result[0].text
