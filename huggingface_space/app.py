"""
Nemotron-Ops-Commander — HuggingFace Spaces Edition.

AI-Powered Incident Response System with 4 specialized SRE agents
and RAG over 30 real-world incidents.

This is the main entry point for the Gradio Space.
"""

from __future__ import annotations

import json
import time
from typing import Tuple

import gradio as gr

from agents import analyze_logs, optimize_performance, triage_incident
from inference import get_client
from rag_engine import get_rag_engine

# ---------------------------------------------------------------------------
# Sample data for quick demos
# ---------------------------------------------------------------------------

SAMPLE_LOGS = """[2026-02-07T10:30:00Z] kubelet: Container payment-api exceeded memory limit: 2048Mi
[2026-02-07T10:30:01Z] kubelet: OOMKilled: Container payment-api in pod payment-api-7d4f8b-x2k9l
[2026-02-07T10:30:02Z] kube-controller: Restarting container payment-api (restart count: 5)
[2026-02-07T10:29:50Z] payment-api: WARN: Heap usage at 94% - GC pause 2300ms
[2026-02-07T10:28:00Z] argocd: Synced application payment-api to revision abc123f (v2.14.0)"""

SAMPLE_INCIDENT_TITLE = "Payment API OOMKilled in production"
SAMPLE_INCIDENT_DESC = """Payment API pods are being OOMKilled after deployment of v2.14.0.
5 restarts in 2 minutes. Customer error rate spiked to 15%.
Heap usage was at 94% before kill. GC pauses exceeding 2 seconds."""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_analysis(result) -> str:
    lines = ["## Log Analysis Results\n"]

    for f in result.findings:
        badge = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(
            f.severity, "⚪"
        )
        lines.append(f"{badge} **{f.severity.upper()}**: {f.summary}")
        lines.append(f"   - Confidence: {f.confidence:.0%}")
        if f.evidence:
            lines.append(f"   - Evidence: {', '.join(f.evidence[:3])}")

    if result.root_cause:
        lines.append(f"\n### Root Cause\n{result.root_cause}")

    if result.recommendations:
        lines.append("\n### Recommendations")
        for i, r in enumerate(result.recommendations, 1):
            lines.append(f"{i}. {r}")

    if not result.findings and result.raw_text:
        lines.append("\n### Model Response")
        lines.append(result.raw_text)

    lines.append(f"\n---\n*Inference latency: {result.latency_ms:.0f}ms*")
    return "\n".join(lines)


def _format_triage(result) -> str:
    badge = {"P0": "🔴", "P1": "🟠", "P2": "🟡", "P3": "🔵", "P4": "⚪"}.get(
        result.priority, "❓"
    )
    lines = [
        "## Incident Triage Results\n",
        f"**Priority:** {badge} {result.priority}",
        f"**Impact:** {result.impact}",
    ]
    if result.likely_services:
        lines.append(f"**Affected services:** {', '.join(result.likely_services)}")
    if result.next_steps:
        lines.append("\n### Next Steps")
        for i, s in enumerate(result.next_steps, 1):
            lines.append(f"{i}. {s}")

    if not result.next_steps and result.raw_text and result.priority == "P3" and result.impact == "unknown":
        lines.append("\n### Model Response")
        lines.append(result.raw_text)

    lines.append(f"\n---\n*Inference latency: {result.latency_ms:.0f}ms*")
    return "\n".join(lines)


def _format_optimization(result) -> str:
    lines = [
        "## Optimization Recommendations\n",
        f"**Bottleneck:** {result.bottleneck}",
        f"**Severity:** {result.severity}",
    ]
    if result.recommendations:
        lines.append("\n### Actions")
        for i, r in enumerate(result.recommendations, 1):
            lines.append(f"{i}. **{r.action}**")
            if r.rationale:
                lines.append(f"   - Rationale: {r.rationale}")
            lines.append(f"   - Risk: {r.risk}")

    # If parsing produced no useful structured data, show the raw model
    # response so the user still sees something meaningful.
    if not result.recommendations and result.raw_text and result.bottleneck in ("unknown", "see analysis below"):
        lines.append("\n### Model Response")
        lines.append(result.raw_text)

    lines.append(f"\n---\n*Inference latency: {result.latency_ms:.0f}ms*")
    return "\n".join(lines)


def _format_rag(results, latency_ms: float) -> str:
    lines = ["## Knowledge Base Results\n"]

    if not results:
        lines.append("*No matching incidents found.*")
    else:
        for hit in results:
            p = hit.payload
            lines.append(
                f"- **[{p.get('source', '?')}]** {p.get('title', hit.id)} "
                f"(score: {hit.score:.2f}, severity: {p.get('severity', '?')})"
            )
            if p.get("resolution"):
                lines.append(f"  - Resolution: {p['resolution'][:300]}")
            if p.get("service"):
                lines.append(f"  - Service: {p['service']}")

    lines.append(f"\n---\n*Search latency: {latency_ms:.0f}ms*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab handlers
# ---------------------------------------------------------------------------

def handle_analyze(log_text: str, system: str, environment: str) -> Tuple[str, str]:
    if not log_text.strip():
        return "Please paste some log entries.", "{}"

    result = analyze_logs(log_text, system=system or "unknown", environment=environment or "production")
    formatted = _format_analysis(result)
    raw = json.dumps(result.model_dump(), indent=2, default=str)
    return formatted, raw


def handle_triage(title: str, description: str, error_rate: float, latency_p99: float) -> Tuple[str, str]:
    if not description.strip():
        return "Please provide an incident description.", "{}"

    result = triage_incident(title, description, error_rate, latency_p99)
    formatted = _format_triage(result)
    raw = json.dumps(result.model_dump(), indent=2, default=str)
    return formatted, raw


def handle_optimize(cpu: float, memory: float, gpu: float, service: str, context: str) -> Tuple[str, str]:
    result = optimize_performance(cpu, memory, gpu, service=service or "unknown", context=context)
    formatted = _format_optimization(result)
    raw = json.dumps(result.model_dump(), indent=2, default=str)
    return formatted, raw


def handle_rag(query: str, top_k: int) -> Tuple[str, str]:
    if not query.strip():
        return "Please enter a search query.", "{}"

    engine = get_rag_engine()
    start = time.time()
    results = engine.search(query, top_k=int(top_k))
    latency_ms = (time.time() - start) * 1000

    formatted = _format_rag(results, latency_ms)
    raw = json.dumps([r.model_dump() for r in results], indent=2, default=str)
    return formatted, raw


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    # Initialize RAG engine on startup (indexes 30 incidents)
    engine = get_rag_engine()
    incident_count = engine.count()

    # Get active model name
    try:
        client = get_client()
        model_name = client.get_active_model()
    except Exception:
        model_name = "initializing..."

    theme = gr.themes.Base(primary_hue="green", neutral_hue="slate")

    with gr.Blocks(title="Nemotron-Ops-Commander", theme=theme) as demo:
        gr.Markdown(
            "# Nemotron-Ops-Commander\n"
            "**AI-Powered Incident Response** for SRE Teams\n\n"
            f"*Model: `{model_name}` "
            f"| RAG: {incident_count} indexed incidents "
            f"| Embeddings: all-MiniLM-L6-v2*"
        )

        with gr.Tabs():
            # ── Tab 1: Log Analysis ──────────────────────────────
            with gr.TabItem("Log Analysis"):
                gr.Markdown("Paste Kubernetes/application logs for AI-powered root cause analysis.")
                with gr.Row():
                    with gr.Column(scale=1):
                        log_input = gr.Textbox(
                            lines=12,
                            label="Log Entries",
                            placeholder="Paste log lines here...",
                            value=SAMPLE_LOGS,
                        )
                        with gr.Row():
                            system_input = gr.Textbox(value="payment-api", label="Service")
                            env_input = gr.Textbox(value="production", label="Environment")
                        analyze_btn = gr.Button("Analyze Logs", variant="primary")
                    with gr.Column(scale=1):
                        analysis_output = gr.Markdown(label="Analysis")
                        with gr.Accordion("Raw JSON", open=False):
                            analysis_raw = gr.Code(label="Raw JSON", language="json")

                analyze_btn.click(
                    fn=handle_analyze,
                    inputs=[log_input, system_input, env_input],
                    outputs=[analysis_output, analysis_raw],
                )

            # ── Tab 2: Incident Triage ───────────────────────────
            with gr.TabItem("Incident Triage"):
                gr.Markdown("Describe an incident for AI-powered severity classification and action plan.")
                with gr.Row():
                    with gr.Column(scale=1):
                        title_input = gr.Textbox(
                            label="Incident Title",
                            value=SAMPLE_INCIDENT_TITLE,
                        )
                        desc_input = gr.Textbox(
                            lines=6,
                            label="Description",
                            value=SAMPLE_INCIDENT_DESC,
                        )
                        with gr.Row():
                            error_rate = gr.Slider(
                                0, 1, value=0.15, step=0.01, label="Error Rate"
                            )
                            latency_input = gr.Slider(
                                0, 30000, value=8500, step=100, label="p99 Latency (ms)"
                            )
                        triage_btn = gr.Button("Triage Incident", variant="primary")
                    with gr.Column(scale=1):
                        triage_output = gr.Markdown(label="Triage Result")
                        with gr.Accordion("Raw JSON", open=False):
                            triage_raw = gr.Code(label="Raw JSON", language="json")

                triage_btn.click(
                    fn=handle_triage,
                    inputs=[title_input, desc_input, error_rate, latency_input],
                    outputs=[triage_output, triage_raw],
                )

            # ── Tab 3: Performance Optimizer ─────────────────────
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
                        with gr.Accordion("Raw JSON", open=False):
                            opt_raw = gr.Code(label="Raw JSON", language="json")

                opt_btn.click(
                    fn=handle_optimize,
                    inputs=[cpu_slider, mem_slider, gpu_slider, svc_input, ctx_input],
                    outputs=[opt_output, opt_raw],
                )

            # ── Tab 4: Knowledge Search (RAG) ────────────────────
            with gr.TabItem("Knowledge Search"):
                gr.Markdown(
                    "Semantic search over **30 real-world incidents** from Kubernetes, "
                    "AWS, Azure, StackOverflow, and GitHub."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        rag_query = gr.Textbox(
                            label="Search Query",
                            value="pod OOMKilled memory limit kubernetes",
                        )
                        rag_topk = gr.Slider(
                            1, 10, value=5, step=1, label="Top K Results"
                        )
                        rag_btn = gr.Button("Search Knowledge Base", variant="primary")
                    with gr.Column(scale=1):
                        rag_output = gr.Markdown(label="Results")
                        with gr.Accordion("Raw JSON", open=False):
                            rag_raw = gr.Code(label="Raw JSON", language="json")

                rag_btn.click(
                    fn=handle_rag,
                    inputs=[rag_query, rag_topk],
                    outputs=[rag_output, rag_raw],
                )

        gr.Markdown(
            "---\n"
            "*Powered by [NVIDIA Nemotron](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) "
            "| ChromaDB RAG | sentence-transformers | Built for GTC Golden Ticket*\n\n"
            "**Local GPU inference** when running on T4/A10 Spaces. "
            "Set `HF_TOKEN` as a Space secret for gated model access."
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
