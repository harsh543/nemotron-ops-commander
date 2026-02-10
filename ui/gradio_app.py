"""
Gradio demo UI for Nemotron-Ops-Commander.

Features:
- Log Analysis tab with sample logs
- Incident Triage tab
- Performance Optimizer tab
- RAG Knowledge Search tab
- Live performance metrics
"""

from __future__ import annotations

import json
import os
import time
from typing import Tuple

import gradio as gr
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "change-me")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
TIMEOUT = 60

# ---------------------------------------------------------------------------
# Sample data for quick demos
# ---------------------------------------------------------------------------

SAMPLE_LOGS = """[2026-02-07T10:30:00Z] kubelet: Container payment-api exceeded memory limit: 2048Mi
[2026-02-07T10:30:01Z] kubelet: OOMKilled: Container payment-api in pod payment-api-7d4f8b-x2k9l
[2026-02-07T10:30:02Z] kube-controller: Restarting container payment-api (restart count: 5)
[2026-02-07T10:29:50Z] payment-api: WARN: Heap usage at 94% - GC pause 2300ms
[2026-02-07T10:28:00Z] argocd: Synced application payment-api to revision abc123f (v2.14.0)"""

SAMPLE_INCIDENT = """Payment API pods are being OOMKilled after deployment of v2.14.0.
5 restarts in 2 minutes. Customer error rate spiked to 15%.
Heap usage was at 94% before kill. GC pauses exceeding 2 seconds."""

# ── Additional sample scenarios for demo / judging ──────────────────────

SAMPLE_LOGS_DNS = """\
[2026-02-07T12:05:11Z] app: ERROR: Temporary failure in name resolution: api.stripe.com
[2026-02-07T12:05:12Z] app: ERROR: Temporary failure in name resolution: api.stripe.com
[2026-02-07T12:05:15Z] coredns: plugin/errors: 2 api.stripe.com. A: read udp 10.0.1.2:38122->10.0.0.10:53: i/o timeout
[2026-02-07T12:05:20Z] kubelet: OOMKilled: Container coredns in pod coredns-6f8d
[2026-02-07T12:05:25Z] coredns: [FATAL] plugin/forward: too many open files"""

SAMPLE_LOGS_AUTH = """\
[2026-02-07T09:10:03Z] api-gateway: WARN: JWT signature verification failed
[2026-02-07T09:10:04Z] api-gateway: ERROR: 401 Unauthorized for /v1/orders
[2026-02-07T09:10:05Z] auth-service: INFO: JWKS cache refresh succeeded
[2026-02-07T09:10:06Z] api-gateway: WARN: kid=old-key-2023 not found in JWKS
[2026-02-07T09:10:07Z] api-gateway: ERROR: 401 Unauthorized for /v1/payments"""

SAMPLE_LOGS_GPU = """\
[2026-02-07T08:15:00Z] kubelet: Container ml-trainer failed to start: OCI runtime create failed
[2026-02-07T08:15:01Z] nvidia-device-plugin: Failed to initialize NVML: could not load NVML library
[2026-02-07T08:15:02Z] ml-trainer: CUDA initialization error: CUDA driver version is insufficient for CUDA runtime version
[2026-02-07T08:15:03Z] kubelet: Back-off restarting failed container ml-trainer in pod training-job-8x4a2
[2026-02-07T08:14:50Z] kured: Reboot required detected on node gpu-worker-03 (kernel 5.15.0-94)"""


# ---------------------------------------------------------------------------
# API call helpers
# ---------------------------------------------------------------------------

def _call_api(endpoint: str, payload: dict) -> Tuple[str, str]:
    """Call API and return (formatted_result, raw_json)."""
    start = time.time()
    try:
        with httpx.Client(base_url=API_URL, headers=HEADERS, timeout=TIMEOUT) as client:
            resp = client.post(endpoint, json=payload)
            latency = (time.time() - start) * 1000

            if resp.status_code != 200:
                return f"Error: HTTP {resp.status_code}\n{resp.text}", "{}"

            data = resp.json()
            data["_demo_latency_ms"] = round(latency, 1)
            raw = json.dumps(data, indent=2, default=str)
            return _format_result(endpoint, data), raw

    except httpx.ConnectError:
        return ("**Connection failed.** Is the API server running?\n\n"
                "Start it with: `uvicorn api.main:app --port 8000`"), "{}"
    except Exception as e:
        return f"Error: {e}", "{}"


def _format_result(endpoint: str, data: dict) -> str:
    """Format API response as readable markdown."""
    lines = []
    latency = data.pop("_demo_latency_ms", 0)

    if "/analyze" in endpoint:
        lines.append("## Log Analysis Results\n")
        for f in data.get("findings", []):
            sev = f.get("severity", "?")
            badge = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(sev, "⚪")
            lines.append(f"{badge} **{sev.upper()}**: {f.get('summary', 'N/A')}")
            lines.append(f"   - Confidence: {f.get('confidence', 0):.0%}")
            if f.get("evidence"):
                lines.append(f"   - Evidence: {', '.join(f['evidence'][:3])}")
        if data.get("root_cause"):
            lines.append(f"\n### Root Cause\n{data['root_cause']}")
        if data.get("recommendations"):
            lines.append("\n### Recommendations")
            for i, r in enumerate(data["recommendations"], 1):
                lines.append(f"{i}. {r}")

    elif "/triage" in endpoint:
        lines.append("## Incident Triage Results\n")
        p = data.get("priority", "?")
        badge = {"P0": "🔴", "P1": "🟠", "P2": "🟡", "P3": "🔵", "P4": "⚪"}.get(p, "❓")
        lines.append(f"**Priority:** {badge} {p}")
        lines.append(f"**Impact:** {data.get('impact', 'N/A')}")
        services = data.get("likely_services", [])
        if services:
            lines.append(f"**Affected services:** {', '.join(services)}")
        if data.get("next_steps"):
            lines.append("\n### Next Steps")
            for i, s in enumerate(data["next_steps"], 1):
                lines.append(f"{i}. {s}")

    elif "/optimize" in endpoint:
        lines.append("## Optimization Recommendations\n")
        lines.append(f"**Bottleneck:** {data.get('bottleneck', 'N/A')}")
        lines.append(f"**Severity:** {data.get('severity', 'N/A')}")
        if data.get("recommendations"):
            lines.append("\n### Actions")
            for i, r in enumerate(data["recommendations"], 1):
                if isinstance(r, dict):
                    lines.append(f"{i}. **{r.get('action', 'N/A')}**")
                    lines.append(f"   - Rationale: {r.get('rationale', '')}")
                    lines.append(f"   - Risk: {r.get('risk', '?')}")
                else:
                    lines.append(f"{i}. {r}")

    elif "/rag" in endpoint:
        lines.append("## Knowledge Base Results\n")
        for hit in data.get("results", []):
            payload = hit.get("payload", {})
            lines.append(
                f"- **[{payload.get('source', '?')}]** {payload.get('title', hit.get('id', '?'))} "
                f"(score: {hit.get('score', 0):.2f})"
            )
            if payload.get("resolution"):
                lines.append(f"  - Resolution: {payload['resolution'][:200]}...")

    lines.append(f"\n---\n*Inference latency: {data.get('latency_ms', latency):.0f}ms "
                 f"| Round-trip: {latency:.0f}ms*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab handlers
# ---------------------------------------------------------------------------

def analyze_logs(log_text: str, system: str, environment: str) -> Tuple[str, str]:
    logs = []
    for line in log_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        logs.append({"timestamp": "now", "source": system or "demo", "message": line, "labels": {}})

    if not logs:
        return "Please paste some log entries.", "{}"

    return _call_api("/analyze/", {
        "logs": logs,
        "system": system or "unknown",
        "environment": environment or "production",
    })


def triage_incident(title: str, description: str, error_rate: float, latency_p99: float) -> Tuple[str, str]:
    if not description:
        return "Please provide an incident description.", "{}"

    return _call_api("/triage/", {
        "incident_id": f"INC-DEMO-{int(time.time()) % 10000:04d}",
        "title": title or "Untitled incident",
        "description": description,
        "metrics": {"error_rate": error_rate, "p99_latency_ms": latency_p99},
    })


def optimize_perf(cpu: float, memory: float, gpu: float, service: str, context: str) -> Tuple[str, str]:
    return _call_api("/optimize/", {
        "metrics": {"cpu": cpu, "memory": memory, "gpu": gpu},
        "service": service or "unknown",
        "context": context,
    })


def search_knowledge(query: str, top_k: int) -> Tuple[str, str]:
    if not query:
        return "Please enter a search query.", "{}"
    return _call_api("/rag/", {"query": query, "top_k": int(top_k)})


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue="green",
        neutral_hue="slate",
    )

    with gr.Blocks(title="Nemotron-Ops-Commander", theme=theme) as demo:
        gr.Markdown(
            "# Nemotron-Ops-Commander\n"
            "**AI-Powered Incident Response** using NVIDIA Nemotron with SGLang optimizations\n\n"
            f"*API: `{API_URL}` | Model: Nemotron-Mini-4B-Instruct*"
        )

        with gr.Tabs():
            # Tab 1: Log Analysis
            with gr.TabItem("Log Analysis"):
                gr.Markdown("Paste Kubernetes/application logs for AI-powered root cause analysis.")
                with gr.Row():
                    with gr.Column(scale=1):
                        log_input = gr.Textbox(
                            lines=12, label="Log Entries",
                            placeholder="Paste log lines here...",
                            value=SAMPLE_LOGS,
                        )
                        with gr.Row():
                            system_input = gr.Textbox(value="payment-api", label="Service")
                            env_input = gr.Textbox(value="production", label="Environment")
                        analyze_btn = gr.Button("Analyze Logs", variant="primary")
                    with gr.Column(scale=1):
                        analysis_output = gr.Markdown(label="Analysis")
                        analysis_raw = gr.Code(label="Raw JSON", language="json", visible=False)

                analyze_btn.click(
                    fn=analyze_logs,
                    inputs=[log_input, system_input, env_input],
                    outputs=[analysis_output, analysis_raw],
                )

                gr.Examples(
                    examples=[
                        [SAMPLE_LOGS_DNS, "coredns", "production"],
                        [SAMPLE_LOGS_AUTH, "api-gateway", "production"],
                        [SAMPLE_LOGS_GPU, "ml-trainer", "production"],
                    ],
                    inputs=[log_input, system_input, env_input],
                    label="More sample scenarios (click to load)",
                )

            # Tab 2: Incident Triage
            with gr.TabItem("Incident Triage"):
                gr.Markdown("Describe an incident for AI-powered severity classification and action plan.")
                with gr.Row():
                    with gr.Column(scale=1):
                        title_input = gr.Textbox(
                            label="Incident Title",
                            value="Payment API OOMKilled in production",
                        )
                        desc_input = gr.Textbox(
                            lines=6, label="Description",
                            value=SAMPLE_INCIDENT,
                        )
                        with gr.Row():
                            error_rate = gr.Slider(0, 1, value=0.15, step=0.01, label="Error Rate")
                            latency_input = gr.Slider(0, 30000, value=8500, step=100, label="p99 Latency (ms)")
                        triage_btn = gr.Button("Triage Incident", variant="primary")
                    with gr.Column(scale=1):
                        triage_output = gr.Markdown(label="Triage Result")
                        triage_raw = gr.Code(label="Raw JSON", language="json", visible=False)

                triage_btn.click(
                    fn=triage_incident,
                    inputs=[title_input, desc_input, error_rate, latency_input],
                    outputs=[triage_output, triage_raw],
                )

                gr.Examples(
                    examples=[
                        [
                            "DNS resolution failures across cluster",
                            "Multiple services report 'Temporary failure in name resolution.' "
                            "CoreDNS pods are OOMKilled and restart every 3\u20135 minutes. "
                            "External API calls (Stripe, Slack) fail intermittently (~20%). "
                            "Recent change: increased ndots setting to 5 in base image. "
                            "Incident started after a config rollout to all namespaces.",
                            0.20,
                            6000,
                        ],
                        [
                            "GPU training jobs failing with CUDA errors",
                            "All pods requesting nvidia.com/gpu fail with CUDA init errors. "
                            "nvidia-smi fails with 'couldn't communicate with driver.' "
                            "Kernel auto-updated last night; nodes rebooted by kured. "
                            "8 A100 nodes affected; training pipeline down for 2 hours.",
                            0.05,
                            12000,
                        ],
                        [
                            "Helm upgrade fails with another operation in progress",
                            "Helm upgrade --install for api-gateway chart failed with "
                            "'another operation (install/upgrade/rollback) is in progress'. "
                            "No other Helm operations running. Last release revision shows "
                            "'pending-upgrade' from a failed deploy 3 days ago. ArgoCD retried "
                            "5 times with same error. Manual helm rollback also fails. "
                            "API gateway running on old revision, cannot deploy critical security patches.",
                            0.02,
                            4500,
                        ],
                    ],
                    inputs=[title_input, desc_input, error_rate, latency_input],
                    label="More sample incidents (click to load)",
                )

            # Tab 3: Performance Optimizer
            with gr.TabItem("Performance Optimizer"):
                gr.Markdown("Input system metrics for AI-powered optimization recommendations.")
                with gr.Row():
                    with gr.Column(scale=1):
                        cpu_slider = gr.Slider(0, 100, value=85, label="CPU Usage %")
                        mem_slider = gr.Slider(0, 100, value=90, label="Memory Usage %")
                        gpu_slider = gr.Slider(0, 100, value=45, label="GPU Usage %")
                        svc_input = gr.Textbox(value="ml-inference", label="Service Name")
                        ctx_input = gr.Textbox(
                            value="Production ML inference service running on K8s with A10 GPUs",
                            label="Context",
                        )
                        opt_btn = gr.Button("Analyze & Optimize", variant="primary")
                    with gr.Column(scale=1):
                        opt_output = gr.Markdown(label="Recommendations")
                        opt_raw = gr.Code(label="Raw JSON", language="json", visible=False)

                opt_btn.click(
                    fn=optimize_perf,
                    inputs=[cpu_slider, mem_slider, gpu_slider, svc_input, ctx_input],
                    outputs=[opt_output, opt_raw],
                )

                gr.Examples(
                    examples=[
                        [
                            92, 88, 35, "ml-inference",
                            "Production inference service on K8s with A10 GPUs; batch size 16; "
                            "latency SLO 120ms; p99 currently 350ms; autoscaling limited to 4 replicas.",
                        ],
                        [
                            65, 97, 10, "order-processing",
                            "Java service with 2Gi memory limit and -Xmx1536m. "
                            "Seeing OOMKills during peak traffic; thread pool size 200; "
                            "Netty direct buffers enabled.",
                        ],
                        [
                            40, 55, 85, "training-orchestrator",
                            "Distributed training jobs; GPU utilization high but throughput flat. "
                            "Network RX/TX spikes; NCCL timeouts; running on mixed instance types.",
                        ],
                    ],
                    inputs=[cpu_slider, mem_slider, gpu_slider, svc_input, ctx_input],
                    label="More sample scenarios (click to load)",
                )

            # Tab 4: Knowledge Search (RAG)
            with gr.TabItem("Knowledge Search"):
                gr.Markdown(
                    "Semantic search over 30 real-world incidents from K8s, AWS, Azure, "
                    "StackOverflow, and GitHub."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        rag_query = gr.Textbox(
                            label="Search Query",
                            value="pod OOMKilled memory limit kubernetes",
                        )
                        rag_topk = gr.Slider(1, 10, value=5, step=1, label="Top K Results")
                        rag_btn = gr.Button("Search Knowledge Base", variant="primary")
                    with gr.Column(scale=1):
                        rag_output = gr.Markdown(label="Results")
                        rag_raw = gr.Code(label="Raw JSON", language="json", visible=False)

                rag_btn.click(
                    fn=search_knowledge,
                    inputs=[rag_query, rag_topk],
                    outputs=[rag_output, rag_raw],
                )

                gr.Examples(
                    examples=[
                        ["CoreDNS OOM ndots 5 dns failures", 5],
                        ["EKS autoscaler max size reached pending pods", 5],
                        ["CUDA driver version insufficient after kernel update", 5],
                        ["Helm pending-upgrade another operation in progress", 5],
                        ["AKS nodepool upgrade failed poddisruptionbudget", 5],
                    ],
                    inputs=[rag_query, rag_topk],
                    label="Sample queries (click to load)",
                )

        gr.Markdown(
            "---\n"
            "*Powered by [NVIDIA Nemotron](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) "
            "| SGLang optimized | ChromaDB RAG | Built for GTC Golden Ticket*"
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
