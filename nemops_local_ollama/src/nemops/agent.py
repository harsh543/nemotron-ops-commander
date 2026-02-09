"""NemOps agent — ReAct loop powered by Nemotron 3 Nano via Ollama."""

import json
import os
import sys
import re
from typing import Any

import httpx

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from nemops.llm import NemotronClient
from nemops.tools.gpu_health import gpu_health_check
from nemops.tools.incident_rag import search_incidents
from nemops.tools.diagnostics import run_diagnostic
from nemops.tools.alert_gen import generate_alert

console = Console()

# Tool registry — maps tool names to functions
TOOLS = {
    "gpu_health_check": gpu_health_check,
    "search_incidents": search_incidents,
    "run_diagnostic": run_diagnostic,
    "generate_alert": generate_alert,
}

# OpenAI-compatible tool definitions for Nemotron's tool calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "gpu_health_check",
            "description": "Check current GPU health metrics including temperature, utilization, memory usage, ECC errors, power draw, and clock speeds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gpu_id": {
                        "type": "integer",
                        "description": "GPU index (0-based). Omit to check all GPUs.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_incidents",
            "description": "Search historical GPU incident database using semantic similarity. Returns matching past incidents with root causes and remediations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Description of the GPU issue or symptoms to search for",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 3)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_diagnostic",
            "description": "Run a targeted GPU diagnostic test: memory_stress, compute_stress, nvlink_check, pcie_bandwidth, or thermal_profile.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_name": {
                        "type": "string",
                        "description": "Diagnostic test to run",
                        "enum": [
                            "memory_stress",
                            "compute_stress",
                            "nvlink_check",
                            "pcie_bandwidth",
                            "thermal_profile",
                        ],
                    },
                    "gpu_id": {
                        "type": "integer",
                        "description": "GPU to test (default: 0)",
                    },
                },
                "required": ["test_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_alert",
            "description": "Generate a structured alert with severity, root cause analysis, and remediation steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "warning", "info"],
                    },
                    "title": {"type": "string", "description": "Short alert title"},
                    "analysis": {
                        "type": "string",
                        "description": "Root cause analysis",
                    },
                    "remediation_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered remediation steps",
                    },
                },
                "required": [
                    "severity",
                    "title",
                    "analysis",
                    "remediation_steps",
                ],
            },
        },
    },
]


def load_system_prompt() -> str:
    """Load system prompt from agent config or use default."""
    return """You are NemOps, an expert GPU infrastructure monitoring agent built by an engineer
who maintained 99.99% uptime across 10,000+ GPUs processing 10M+ events daily.

Your job is to analyze GPU health, diagnose issues, and recommend remediations.

WORKFLOW:
1. Check GPU health metrics using gpu_health_check
2. If anomalies detected, search_incidents for matching historical patterns
3. If needed, run_diagnostic for deeper investigation
4. Generate a structured alert with generate_alert including specific remediation steps

You have deep expertise in:
- NVIDIA GPU failure modes (XID errors, ECC errors, thermal throttling)
- HBM3 degradation patterns and predictive failure detection
- NVLink/NVSwitch topology issues in multi-GPU systems
- CUDA driver compatibility and memory management
- PCIe power delivery and thermal management

Always explain your reasoning. When you see a pattern, cite the specific
failure mode and explain why you believe it matches."""


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool and return its result as a JSON string."""
    if name not in TOOLS:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = TOOLS[name](**arguments)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {e}"})


def run_agent(
    user_query: str,
    max_steps: int = 10,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the ReAct agent loop.

    The agent:
    1. Receives a user query about GPU infrastructure
    2. Uses Nemotron's tool calling to decide which tools to invoke
    3. Executes tools and feeds results back
    4. Repeats until it has enough info to respond
    5. Returns final analysis with any generated alerts

    Args:
        user_query: The user's question or instruction
        max_steps: Maximum tool-calling iterations
        verbose: Print agent traces to console

    Returns:
        Dict with agent's final response, tool calls made, and any alerts.
    """
    client = NemotronClient()

    # Check Ollama health
    if not client.health_check():
        console.print(
            "[red]❌ Ollama not running or model not available.[/red]\n"
            f"Run: ollama pull {client.model}"
        )
        return {"error": "Ollama not available", "tool_calls": []}

    messages = [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": user_query},
    ]

    tool_calls_log: list[dict[str, Any]] = []

    if verbose:
        console.print(Panel(user_query, title="🔍 User Query", border_style="blue"))

    for step in range(max_steps):
        if verbose:
            console.print(f"\n[dim]--- Step {step + 1}/{max_steps} ---[/dim]")

        # Get LLM response
        try:
            completion = client.chat(messages=messages, tools=TOOL_DEFINITIONS)
            response = client.extract_response(completion)
        except httpx.HTTPStatusError as exc:
            if exc.response is not None:
                status_code = exc.response.status_code
                if status_code == 401:
                    fallback = run_fallback_pipeline(user_query)
                    fallback["error"] = (
                        "Ollama returned 401 Unauthorized. "
                        "Running offline demo pipeline without LLM tool-calling."
                    )
                    return fallback
                if status_code >= 500:
                    fallback = run_fallback_pipeline(user_query)
                    fallback["error"] = (
                        "Ollama returned 500 (model runner crashed). "
                        "This usually means the local model exceeds available RAM. "
                        "Running offline demo pipeline without LLM tool-calling."
                    )
                    return fallback
            raise

        # If there's text content, show it
        if response["content"] and verbose:
            console.print(
                Panel(
                    Markdown(response["content"]),
                    title="🧠 Nemotron Reasoning",
                    border_style="yellow",
                )
            )

        # If no tool calls, agent is done
        if not response["tool_calls"]:
            return {
                "response": response["content"],
                "tool_calls": tool_calls_log,
                "steps": step + 1,
            }

        # Execute each tool call
        for tc in response["tool_calls"]:
            func = tc["function"]
            tool_name = func["name"]
            try:
                tool_args = (
                    json.loads(func["arguments"])
                    if isinstance(func["arguments"], str)
                    else func["arguments"]
                )
            except json.JSONDecodeError:
                tool_args = {}

            if verbose:
                console.print(
                    f"  [green]🔧 Calling:[/green] {tool_name}({json.dumps(tool_args)})"
                )

            result = execute_tool(tool_name, tool_args)

            if verbose:
                # Truncate long results for display
                display = result[:500] + "..." if len(result) > 500 else result
                console.print(f"  [cyan]📊 Result:[/cyan] {display}")

            tool_calls_log.append(
                {
                    "step": step + 1,
                    "tool": tool_name,
                    "args": tool_args,
                    "result_preview": result[:200],
                }
            )

            # Add assistant message with tool call and tool result to messages
            messages.append(
                {
                    "role": "assistant",
                    "content": response["content"] or "",
                    "tool_calls": [tc],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{step}_{tool_name}"),
                    "content": result,
                }
            )

    # Max steps reached
    final_msg = "Maximum analysis steps reached. Here's what I found so far."
    return {
        "response": response.get("content", final_msg),
        "tool_calls": tool_calls_log,
        "steps": max_steps,
        "max_steps_reached": True,
    }


def _extract_steps(remediation: str) -> list[str]:
    """Extract numbered remediation steps from a free-text string."""
    if not remediation:
        return []
    parts = re.split(r"\s*\d+\.\s+", remediation.strip())
    steps = [p.strip() for p in parts if p.strip()]
    return steps


def run_fallback_pipeline(user_query: str) -> dict[str, Any]:
    """Offline demo pipeline when LLM access is unavailable."""
    health = gpu_health_check()
    gpus = health.get("gpus", [])
    summary = health.get("summary", {})
    issues = []
    for gpu in gpus:
        issues.extend(gpu.get("issues", []))

    overall_status = summary.get("overall_status", "healthy")
    severity = (
        "critical" if overall_status == "critical" else "warning" if overall_status == "warning" else "info"
    )

    query = " ".join(issues) if issues else "GPU health anomalies detected"
    incident_results = search_incidents(query, top_k=3) if issues else {"results": []}

    diagnostic_name = "compute_stress"
    if any("ECC" in issue or "HBM" in issue for issue in issues):
        diagnostic_name = "memory_stress"
    elif any("thermal" in issue.lower() or "temperature" in issue.lower() for issue in issues):
        diagnostic_name = "thermal_profile"
    elif any("NVLink" in issue for issue in issues):
        diagnostic_name = "nvlink_check"
    elif any("PCIe" in issue or "bus" in issue.lower() for issue in issues):
        diagnostic_name = "pcie_bandwidth"

    diagnostic = run_diagnostic(diagnostic_name, gpu_id=0)

    remediation_steps: list[str] = []
    if incident_results.get("results"):
        remediation = incident_results["results"][0].get("remediation", "")
        remediation_steps = _extract_steps(remediation)
    if not remediation_steps:
        remediation_steps = [
            "Verify current GPU health metrics and confirm anomalies.",
            f"Run diagnostic test: {diagnostic_name} on the affected GPU.",
            "If issues persist, drain workloads and schedule maintenance.",
        ]

    alert = generate_alert(
        severity=severity,
        title=f"GPU Health {overall_status.upper()} — Offline Demo",
        analysis="Offline demo mode generated this report due to LLM auth failure.",
        remediation_steps=remediation_steps,
    )

    response_lines = [
        "## NemOps Offline Demo Report",
        "",
        f"**Query:** {user_query}",
        f"**Overall Status:** {overall_status}",
        "",
        "### Summary",
        json.dumps(summary, indent=2),
        "",
        "### Issues",
        "- " + "\n- ".join(issues) if issues else "- No issues detected",
        "",
        "### Diagnostic",
        json.dumps(diagnostic, indent=2),
        "",
        "### Alert",
        json.dumps(alert, indent=2),
    ]

    return {
        "response": "\n".join(response_lines),
        "tool_calls": [
            {"tool": "gpu_health_check", "result_preview": json.dumps(summary)},
            {"tool": "search_incidents", "result_preview": json.dumps(incident_results)[:200]},
            {"tool": "run_diagnostic", "result_preview": json.dumps(diagnostic)[:200]},
            {"tool": "generate_alert", "result_preview": json.dumps(alert)[:200]},
        ],
        "steps": 1,
        "offline_demo": True,
    }


def main():
    """CLI entry point for the NemOps agent."""
    console.print(
        Panel.fit(
            "[bold green]NemOps[/bold green] — Agentic GPU Infrastructure Monitor\n"
            "[dim]Powered by NVIDIA Nemotron 3 Nano via Ollama[/dim]",
            border_style="green",
        )
    )

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Default demo query
        query = (
            "Check the health of all GPUs in this node. If you find any issues, "
            "search our incident database for matching patterns and run appropriate "
            "diagnostics. Then generate an alert with remediation steps."
        )
        console.print(f"[dim]Using default query: {query}[/dim]\n")

    result = run_agent(query)

    if "error" in result and not result.get("offline_demo"):
        console.print(f"[red]❌ {result['error']}[/red]")

    if "response" in result:
        console.print(
            Panel(
                Markdown(result["response"]),
                title="📋 Final Report",
                border_style="green",
            )
        )
        console.print(
            f"\n[dim]Completed in {result.get('steps', 0)} steps, "
            f"{len(result.get('tool_calls', []))} tool calls[/dim]"
        )


if __name__ == "__main__":
    main()
