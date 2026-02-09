"""MCP server — expose NemOps agent tools via Model Context Protocol."""

import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from nemops.tools.gpu_health import gpu_health_check
from nemops.tools.incident_rag import search_incidents
from nemops.tools.diagnostics import run_diagnostic
from nemops.tools.alert_gen import generate_alert

app = Server("nemops")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available NemOps tools."""
    return [
        Tool(
            name="gpu_health_check",
            description="Check GPU health metrics (temperature, utilization, ECC errors, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_id": {"type": "integer", "description": "GPU index (optional)"}
                },
            },
        ),
        Tool(
            name="search_incidents",
            description="Search historical GPU incidents using semantic similarity",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Symptoms to search for"},
                    "top_k": {"type": "integer", "description": "Results to return"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="run_diagnostic",
            description="Run GPU diagnostic: memory_stress, compute_stress, nvlink_check, pcie_bandwidth, thermal_profile",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_name": {"type": "string"},
                    "gpu_id": {"type": "integer"},
                },
                "required": ["test_name"],
            },
        ),
        Tool(
            name="generate_alert",
            description="Generate structured alert with remediation steps",
            inputSchema={
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["critical", "warning", "info"]},
                    "title": {"type": "string"},
                    "analysis": {"type": "string"},
                    "remediation_steps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["severity", "title", "analysis", "remediation_steps"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a NemOps tool."""
    tools = {
        "gpu_health_check": lambda args: gpu_health_check(**args),
        "search_incidents": lambda args: search_incidents(**args),
        "run_diagnostic": lambda args: run_diagnostic(**args),
        "generate_alert": lambda args: generate_alert(**args),
    }

    if name not in tools:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    result = tools[name](arguments)
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


def main():
    """Run the MCP server."""
    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
