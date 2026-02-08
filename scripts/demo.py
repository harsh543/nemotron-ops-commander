"""
End-to-end demo script for Nemotron-Ops-Commander.

Runs through realistic incident scenarios showing all agents in action.
Designed for live demos and contest presentations.

Usage:
    python scripts/demo.py                    # Run all scenarios
    python scripts/demo.py --scenario 1       # Run specific scenario
    python scripts/demo.py --api              # Use API server mode
    python scripts/demo.py --benchmark        # Include perf numbers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Rich console output helpers
# ---------------------------------------------------------------------------

def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"

def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m"

def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m"

def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"

def _cyan(text: str) -> str:
    return f"\033[96m{text}\033[0m"

def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m"

def _severity_color(severity: str) -> str:
    colors = {"P0": _red, "P1": _red, "critical": _red, "high": _red,
              "P2": _yellow, "medium": _yellow, "P3": _cyan, "P4": _dim,
              "low": _cyan, "info": _dim}
    fn = colors.get(severity, str)
    return fn(severity)

def banner(text: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {_bold(text)}")
    print("=" * width)

def section(text: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {_cyan(text)}")
    print(f"{'─' * 50}")

def step(num: int, text: str) -> None:
    print(f"\n  {_bold(f'[Step {num}]')} {text}")

def result_line(key: str, value: Any) -> None:
    print(f"    {_dim(key + ':')} {value}")

def json_block(data: Any) -> None:
    formatted = json.dumps(data, indent=2, default=str)
    for line in formatted.split("\n"):
        print(f"    {_dim(line)}")


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "OOMKilled Production Pod",
        "description": "Payment service pod killed due to memory exhaustion after deployment",
        "logs": [
            {"timestamp": "2026-02-07T10:30:00Z", "source": "kubelet",
             "message": "Container payment-api exceeded memory limit: 2048Mi",
             "labels": {"pod": "payment-api-7d4f8b-x2k9l", "namespace": "production"}},
            {"timestamp": "2026-02-07T10:30:01Z", "source": "kubelet",
             "message": "OOMKilled: Container payment-api in pod payment-api-7d4f8b-x2k9l",
             "labels": {"pod": "payment-api-7d4f8b-x2k9l", "namespace": "production"}},
            {"timestamp": "2026-02-07T10:30:02Z", "source": "kube-controller",
             "message": "Restarting container payment-api (restart count: 5)",
             "labels": {"pod": "payment-api-7d4f8b-x2k9l", "namespace": "production"}},
            {"timestamp": "2026-02-07T10:29:50Z", "source": "payment-api",
             "message": "WARN: Heap usage at 94% - GC pause 2300ms",
             "labels": {"pod": "payment-api-7d4f8b-x2k9l", "namespace": "production"}},
            {"timestamp": "2026-02-07T10:28:00Z", "source": "argocd",
             "message": "Synced application payment-api to revision abc123f (v2.14.0)",
             "labels": {"app": "payment-api"}},
        ],
        "system": "payment-api",
        "environment": "production",
        "triage": {
            "incident_id": "INC-DEMO-001",
            "title": "Payment API OOMKilled in production",
            "description": "Payment API pods are being OOMKilled after deployment of v2.14.0. "
                          "5 restarts in 2 minutes. Customers reporting failed transactions.",
            "metrics": {"error_rate": 0.15, "p99_latency_ms": 8500, "pod_restarts": 5},
        },
        "optimize": {
            "metrics": {"cpu": 45, "memory": 98, "disk_io": 30, "network": 20, "gc_pause_ms": 2300},
            "service": "payment-api",
            "context": "Java Spring Boot service on K8s, 2Gi memory limit, JVM heap not explicitly set",
        },
    },
    {
        "id": 2,
        "name": "Database Connection Storm",
        "description": "PostgreSQL connection pool exhausted causing cascading API failures",
        "logs": [
            {"timestamp": "2026-02-07T14:00:00Z", "source": "user-service",
             "message": "ERROR: could not connect to PostgreSQL: too many connections",
             "labels": {"pod": "user-service-5c8d7a-m3k2l", "namespace": "production"}},
            {"timestamp": "2026-02-07T14:00:01Z", "source": "user-service",
             "message": "ERROR: connection pool exhausted (100/100 active), 47 requests waiting",
             "labels": {"pod": "user-service-5c8d7a-m3k2l", "namespace": "production"}},
            {"timestamp": "2026-02-07T14:00:05Z", "source": "order-service",
             "message": "ERROR: upstream user-service returned 503 Service Unavailable",
             "labels": {"pod": "order-service-8f2a3b-j7n4p", "namespace": "production"}},
            {"timestamp": "2026-02-07T13:59:50Z", "source": "pgbouncer",
             "message": "WARNING: server connection count 200 exceeds max_client_conn 200",
             "labels": {"pod": "pgbouncer-0", "namespace": "database"}},
            {"timestamp": "2026-02-07T13:58:00Z", "source": "hpa",
             "message": "New size: 8; reason: cpu resource utilization above target",
             "labels": {"deployment": "user-service", "namespace": "production"}},
        ],
        "system": "user-service",
        "environment": "production",
        "triage": {
            "incident_id": "INC-DEMO-002",
            "title": "Database connection pool exhaustion",
            "description": "User service unable to acquire database connections. HPA scaled "
                          "pods from 3 to 8, each opening 100 connections, exceeding "
                          "PostgreSQL max_connections (200).",
            "metrics": {"error_rate": 0.35, "p99_latency_ms": 15000, "active_connections": 800},
        },
        "optimize": {
            "metrics": {"cpu": 85, "memory": 60, "db_connections": 800, "db_max_connections": 200, "replicas": 8},
            "service": "user-service + pgbouncer",
            "context": "HPA auto-scaled from 3 to 8 replicas, each with 100-connection pool",
        },
    },
    {
        "id": 3,
        "name": "TLS Certificate Expiry",
        "description": "Ingress TLS cert expired, causing customer-facing HTTPS failures",
        "logs": [
            {"timestamp": "2026-02-07T03:00:00Z", "source": "nginx-ingress",
             "message": "ERROR: SSL certificate for api.example.com expired on 2026-02-06",
             "labels": {"pod": "nginx-ingress-controller-8f4a2c", "namespace": "ingress-system"}},
            {"timestamp": "2026-02-07T03:00:01Z", "source": "nginx-ingress",
             "message": "WARN: Serving with expired certificate - clients will see SSL errors",
             "labels": {"pod": "nginx-ingress-controller-8f4a2c", "namespace": "ingress-system"}},
            {"timestamp": "2026-02-07T03:05:00Z", "source": "cert-manager",
             "message": "ERROR: Failed to renew certificate api-tls: ACME challenge failed DNS-01",
             "labels": {"namespace": "cert-manager"}},
            {"timestamp": "2026-02-07T03:05:01Z", "source": "cert-manager",
             "message": "ERROR: Route53 access denied - IAM role missing route53:ChangeResourceRecordSets",
             "labels": {"namespace": "cert-manager"}},
        ],
        "system": "ingress-system",
        "environment": "production",
        "triage": {
            "incident_id": "INC-DEMO-003",
            "title": "TLS certificate expired for api.example.com",
            "description": "Customer-facing API TLS certificate expired. cert-manager auto-renewal "
                          "failed due to IAM permission missing for Route53 DNS challenge. "
                          "All HTTPS traffic showing certificate errors.",
            "metrics": {"error_rate": 1.0, "https_failures": 12000, "affected_domains": 1},
        },
        "optimize": {
            "metrics": {"cpu": 10, "memory": 20, "error_rate": 100, "cert_days_remaining": -1},
            "service": "ingress-system",
            "context": "cert-manager using DNS-01 ACME challenge via Route53, IAM role expired",
        },
    },
]


# ---------------------------------------------------------------------------
# Direct mode (no API server needed)
# ---------------------------------------------------------------------------

async def run_direct_demo(scenario: Dict[str, Any], include_benchmark: bool = False) -> None:
    """Run demo by calling agents directly (no server required)."""
    from models.nemotron_client import NemotronClient, NemotronConfig
    from agents.log_analyzer import LogAnalyzerAgent
    from agents.incident_triager import IncidentTriagerAgent
    from agents.remediation_suggester import RemediationSuggesterAgent
    from agents.optimizer import OptimizerAgent

    banner(f"Scenario {scenario['id']}: {scenario['name']}")
    print(f"  {_dim(scenario['description'])}")

    # Initialize model
    step(0, "Initializing NVIDIA Nemotron...")
    start = time.time()
    try:
        client = NemotronClient(NemotronConfig())
        init_time = (time.time() - start) * 1000
        result_line("Model", client.config.model_name)
        result_line("Backend", "SGLang" if client.config.use_sglang else "Transformers")
        result_line("Init time", f"{init_time:.0f}ms")
    except Exception as e:
        print(f"\n  {_red('ERROR:')} Could not initialize Nemotron: {e}")
        print(f"  {_yellow('TIP:')} Set NEMOTRON_DEVICE=cpu in .env for CPU-only systems")
        print(f"  {_yellow('TIP:')} Run `python scripts/setup_nemotron.py` first")
        return

    # Step 1: Log Analysis
    step(1, "Analyzing logs with Log Analyzer Agent...")
    print(f"    {_dim('Feeding')} {len(scenario['logs'])} {_dim('log entries to Nemotron...')}")
    agent = LogAnalyzerAgent(client)
    analysis = await agent.run({
        "logs": scenario["logs"],
        "system": scenario["system"],
        "environment": scenario["environment"],
    })
    analysis_data = analysis.model_dump()

    section("Log Analysis Results")
    result_line("Findings", len(analysis_data.get("findings", [])))
    result_line("Root cause", analysis_data.get("root_cause", "N/A"))
    result_line("Latency", f"{analysis_data.get('latency_ms', 0):.0f}ms")
    if analysis_data.get("recommendations"):
        result_line("Recommendations", "")
        for i, rec in enumerate(analysis_data["recommendations"], 1):
            print(f"      {i}. {rec}")

    # Step 2: RAG Retrieval
    step(2, "Searching knowledge base for similar incidents...")
    try:
        from rag.vector_store import ChromaVectorStore
        from rag.embeddings import EmbeddingService
        from rag.retriever import RAGRetriever

        store = ChromaVectorStore()
        embeddings = EmbeddingService()
        retriever = RAGRetriever(embeddings, store)

        rag_start = time.time()
        query = scenario["triage"]["description"]
        rag_result = retriever.query(query, top_k=3)
        rag_time = (time.time() - rag_start) * 1000

        section("Similar Historical Incidents (RAG)")
        result_line("Query latency", f"{rag_time:.0f}ms")
        result_line("Results found", len(rag_result.results))
        for hit in rag_result.results:
            print(f"      - [{hit.payload.get('source', '?')}] "
                  f"{hit.payload.get('title', hit.id)} "
                  f"(score: {hit.score:.2f})")
    except Exception as e:
        print(f"    {_yellow('RAG skipped:')} {e}")
        print(f"    {_dim('Run: python scripts/index_incidents.py')}")

    # Step 3: Incident Triage
    step(3, "Triaging incident with Incident Triager Agent...")
    triager = IncidentTriagerAgent(client)
    triage = await triager.run(scenario["triage"])
    triage_data = triage.model_dump()

    section("Incident Triage Results")
    result_line("Priority", _severity_color(triage_data.get("priority", "?")))
    result_line("Impact", triage_data.get("impact", "N/A"))
    result_line("Likely services", ", ".join(triage_data.get("likely_services", [])))
    result_line("Latency", f"{triage_data.get('latency_ms', 0):.0f}ms")
    if triage_data.get("next_steps"):
        result_line("Next steps", "")
        for i, s in enumerate(triage_data["next_steps"], 1):
            print(f"      {i}. {s}")

    # Step 4: Optimization
    step(4, "Analyzing metrics with Performance Optimizer Agent...")
    optimizer = OptimizerAgent(client)
    opt = await optimizer.run(scenario["optimize"])
    opt_data = opt.model_dump()

    section("Optimization Recommendations")
    result_line("Bottleneck", opt_data.get("bottleneck", "N/A"))
    result_line("Severity", _severity_color(opt_data.get("severity", "?")))
    result_line("Latency", f"{opt_data.get('latency_ms', 0):.0f}ms")
    if opt_data.get("recommendations"):
        result_line("Actions", "")
        for i, rec in enumerate(opt_data["recommendations"], 1):
            action = rec.get("action", rec) if isinstance(rec, dict) else rec
            risk = rec.get("risk", "?") if isinstance(rec, dict) else "?"
            print(f"      {i}. {action} {_dim(f'[risk: {risk}]')}")

    # Step 5: Benchmark (optional)
    if include_benchmark:
        step(5, "Running quick benchmark...")
        latencies = []
        for i in range(5):
            start = time.time()
            await client.generate("Summarize: pod OOMKilled in production")
            latencies.append((time.time() - start) * 1000)

        section("Performance Benchmark (5 iterations)")
        result_line("Mean latency", f"{sum(latencies)/len(latencies):.0f}ms")
        result_line("Min latency", f"{min(latencies):.0f}ms")
        result_line("Max latency", f"{max(latencies):.0f}ms")
        result_line("Backend", "SGLang" if client.config.use_sglang else "Transformers")

    print(f"\n{_green('✓')} Scenario {scenario['id']} complete\n")


# ---------------------------------------------------------------------------
# API mode (requires running server)
# ---------------------------------------------------------------------------

async def run_api_demo(scenario: Dict[str, Any], api_url: str = "http://localhost:8000") -> None:
    """Run demo by calling the FastAPI endpoints."""
    import httpx

    banner(f"Scenario {scenario['id']}: {scenario['name']} (API mode)")
    print(f"  {_dim(scenario['description'])}")
    print(f"  {_dim(f'API: {api_url}')}")

    headers = {"X-API-Key": "change-me", "Content-Type": "application/json"}

    async with httpx.AsyncClient(base_url=api_url, headers=headers, timeout=60) as client:
        # Log Analysis
        step(1, f"POST {api_url}/analyze/")
        payload = {
            "logs": scenario["logs"],
            "system": scenario["system"],
            "environment": scenario["environment"],
        }
        resp = await client.post("/analyze/", json=payload)
        section("Log Analysis Response")
        if resp.status_code == 200:
            json_block(resp.json())
        else:
            print(f"    {_red(f'HTTP {resp.status_code}')}: {resp.text}")

        # Triage
        step(2, f"POST {api_url}/triage/")
        resp = await client.post("/triage/", json=scenario["triage"])
        section("Triage Response")
        if resp.status_code == 200:
            json_block(resp.json())
        else:
            print(f"    {_red(f'HTTP {resp.status_code}')}: {resp.text}")

        # Optimize
        step(3, f"POST {api_url}/optimize/")
        resp = await client.post("/optimize/", json=scenario["optimize"])
        section("Optimization Response")
        if resp.status_code == 200:
            json_block(resp.json())
        else:
            print(f"    {_red(f'HTTP {resp.status_code}')}: {resp.text}")

        # RAG
        step(4, f"POST {api_url}/rag/")
        resp = await client.post("/rag/", json={"query": scenario["triage"]["description"], "top_k": 3})
        section("RAG Results")
        if resp.status_code == 200:
            json_block(resp.json())
        else:
            print(f"    {_red(f'HTTP {resp.status_code}')}: {resp.text}")

    print(f"\n{_green('✓')} Scenario {scenario['id']} complete (API mode)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain(args: argparse.Namespace) -> None:
    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s["id"] == args.scenario]
        if not scenarios:
            print(f"{_red('Error:')} Scenario {args.scenario} not found (available: 1-{len(SCENARIOS)})")
            return

    banner("Nemotron-Ops-Commander Demo")
    print(f"  {_dim('AI-Powered Incident Response using NVIDIA Nemotron')}")
    print(f"  {_dim(f'Scenarios: {len(scenarios)} | Mode: {'API' if args.api else 'Direct'} | Benchmark: {args.benchmark}')}")

    for scenario in scenarios:
        if args.api:
            await run_api_demo(scenario, api_url=args.api_url)
        else:
            await run_direct_demo(scenario, include_benchmark=args.benchmark)

    banner("Demo Complete")
    print(f"  {_green('All scenarios finished successfully.')}")
    print(f"  {_dim('Powered by NVIDIA Nemotron + SGLang')}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nemotron-Ops-Commander Demo")
    parser.add_argument("--scenario", type=int, help="Run specific scenario (1-3)")
    parser.add_argument("--api", action="store_true", help="Use API server mode")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--benchmark", action="store_true", help="Include performance benchmark")
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
