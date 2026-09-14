"""
Nemotron-Ops-Commander — HuggingFace Spaces Edition.

AI-Powered Incident Response System with 4 specialized SRE agents
and RAG over 30 real-world incidents.

This is the main entry point for the Gradio Space.
"""

from __future__ import annotations

import spaces
import json
import os
import time
import uuid
from typing import Tuple

import gradio as gr

from agents import analyze_logs, optimize_performance, triage_incident
from inference import get_client
from rag_engine import EMBEDDING_MODEL, get_rag_engine

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

@spaces.GPU(duration=120)
def handle_analyze(log_text: str, system: str, environment: str) -> Tuple[str, str]:
    """Analyze service logs and return a readable diagnosis plus structured JSON."""
    if not log_text.strip():
        return "Please paste some log entries.", "{}"

    result = analyze_logs(log_text, system=system or "unknown", environment=environment or "production")
    formatted = _format_analysis(result)
    raw = json.dumps(result.model_dump(), indent=2, default=str)
    return formatted, raw


@spaces.GPU(duration=120)
def handle_triage(title: str, description: str, error_rate: float, latency_p99: float) -> Tuple[str, str]:
    """Classify an incident and return impact, affected services, and next steps."""
    if not description.strip():
        return "Please provide an incident description.", "{}"

    result = triage_incident(title, description, error_rate, latency_p99)
    formatted = _format_triage(result)
    raw = json.dumps(result.model_dump(), indent=2, default=str)
    return formatted, raw


@spaces.GPU(duration=120)
def handle_optimize(cpu: float, memory: float, gpu: float, service: str, context: str) -> Tuple[str, str]:
    """Recommend performance improvements for the supplied service metrics."""
    result = optimize_performance(cpu, memory, gpu, service=service or "unknown", context=context)
    formatted = _format_optimization(result)
    raw = json.dumps(result.model_dump(), indent=2, default=str)
    return formatted, raw


@spaces.GPU(duration=120)
def handle_rag(query: str, top_k: int) -> Tuple[str, str]:
    """Search the bundled incident knowledge base with NVIDIA embeddings."""
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
# Pro tier: concurrent multi-incident triage (Subscriptions / RevenueCat
# track). Free tier triages one incident at a time on this Space's shared
# GPU; Pro (unlocked via a RevenueCat Web Billing purchase, Test Store --
# no real money) fans out to Nebius Token Factory and triages several at
# once, demonstrating the concurrency a shared free-tier GPU can't offer.
# ---------------------------------------------------------------------------

PRO_SAMPLE_INCIDENTS = [
    ("Payment API OOMKilled", "5 restarts in 2 minutes, heap at 94% before kill.", 0.15, 5200),
    ("CoreDNS resolution failures", "Intermittent DNS timeouts across the cluster.", 0.08, 2100),
    ("Auth service 401 spike", "JWKS cache stale, old key ID rejected.", 0.22, 800),
]


def new_app_user_id() -> str:
    return str(uuid.uuid4())


def render_purchase_widget(app_user_id: str) -> str:
    """Rebuilt per-session (via demo.load) so the embedded JS gets this
    session's own app_user_id baked in directly, rather than trying to
    thread a Gradio State value into a static HTML string after the fact."""
    public_key = os.environ.get("REVENUECAT_PUBLIC_API_KEY", "")
    package_id = os.environ.get("REVENUECAT_PACKAGE_ID", "pro_monthly")
    if not public_key:
        return "<i>REVENUECAT_PUBLIC_API_KEY not configured on this Space yet.</i>"
    return f"""
    <div>
      <button id="rc-buy-btn" style="padding:8px 16px;font-weight:600;">Buy Pro (Test Store)</button>
      <span id="rc-status" style="margin-left:10px;"></span>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/@revenuecat/purchases-js@1.60.1/dist/Purchases.umd.js"></script>
    <script>
    (function() {{
      const appUserId = {json.dumps(app_user_id)};
      const publicKey = {json.dumps(public_key)};
      const packageId = {json.dumps(package_id)};
      const statusEl = document.getElementById('rc-status');

      document.getElementById('rc-buy-btn').addEventListener('click', async () => {{
        statusEl.textContent = 'Loading offerings...';
        try {{
          const purchases = Purchases.configure({{apiKey: publicKey, appUserId: appUserId}});
          const offerings = await purchases.getOfferings();
          const pkg = (offerings.current && offerings.current.availablePackages.find(p => p.identifier === packageId))
            || (offerings.current && offerings.current.availablePackages[0]);
          if (!pkg) {{ statusEl.textContent = 'No package configured in RevenueCat yet.'; return; }}
          statusEl.textContent = 'Opening Test Store checkout...';
          const result = await purchases.purchase({{rcPackage: pkg}});
          const active = Object.keys(result.customerInfo.entitlements.active);
          statusEl.textContent = active.length ? 'Purchased! Click "Check Pro status" below.' : 'Purchase did not activate an entitlement.';
        }} catch (err) {{
          statusEl.textContent = 'Error: ' + (err && err.message ? err.message : err);
        }}
      }});
    }})();
    </script>
    """


def check_pro_status(app_user_id: str):
    from revenuecat_client import is_pro

    unlocked = is_pro(app_user_id)
    return (
        gr.update(visible=unlocked),
        gr.update(visible=not unlocked),
    )


def handle_concurrent_triage(app_user_id: str) -> str:
    from concurrent.futures import ThreadPoolExecutor

    from revenuecat_client import is_pro

    if not is_pro(app_user_id):
        return "Pro entitlement not active -- purchase required to run concurrent triage."

    start = time.time()
    with ThreadPoolExecutor(max_workers=len(PRO_SAMPLE_INCIDENTS)) as pool:
        results = list(pool.map(lambda args: triage_incident(*args), PRO_SAMPLE_INCIDENTS))
    wall_ms = (time.time() - start) * 1000

    lines = [f"**{len(results)} incidents triaged concurrently in {wall_ms:.0f}ms wall-clock**\n"]
    sequential_estimate = sum(r.latency_ms for r in results)
    lines.append(
        f"(sequential would take ~{sequential_estimate:.0f}ms -- "
        f"{sequential_estimate / wall_ms:.1f}x speedup from running in parallel)\n"
    )
    for (title, *_rest), result in zip(PRO_SAMPLE_INCIDENTS, results):
        lines.append(f"- **{title}** — {result.priority}, {result.latency_ms:.0f}ms — {result.impact}")
    return "\n".join(lines)


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

    embedding_label = EMBEDDING_MODEL.split("/")[-1]

    with gr.Blocks(title="Nemotron-Ops-Commander") as demo:
        gr.Markdown(
            "# Nemotron-Ops-Commander\n"
            "**AI-Powered Incident Response** for SRE Teams\n\n"
            f"*Model: `{model_name}` "
            f"| RAG: {incident_count} indexed incidents "
            f"| Embeddings: {embedding_label}*"
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
                    api_name="analyze_logs",
                    concurrency_limit=1,
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
                    api_name="triage_incident",
                    concurrency_limit=1,
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
                    api_name="optimize_performance",
                    concurrency_limit=1,
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
                    api_name="search_incidents",
                    concurrency_limit=1,
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

            # ── Tab 5: Pro -- Concurrent Triage (RevenueCat + Nebius) ────
            with gr.TabItem("Pro: Concurrent Triage"):
                app_user_id_state = gr.State("")

                gr.Markdown(
                    "Free tier triages one incident at a time. **Pro** fans out to "
                    "**Nebius Token Factory** and triages several incidents "
                    "concurrently -- the throughput a shared free-tier GPU can't give you. "
                    "This is a RevenueCat **Test Store** purchase: no real money moves."
                )

                purchase_widget = gr.HTML()

                check_pro_btn = gr.Button("Check Pro status / Unlock")

                with gr.Group(visible=False) as pro_group:
                    gr.Markdown("**Pro unlocked** — 3 sample incidents, triaged concurrently via Nebius Token Factory:")
                    run_concurrent_btn = gr.Button("Run concurrent triage", variant="primary")
                    concurrent_output = gr.Markdown()

                with gr.Group(visible=True) as locked_group:
                    gr.Markdown("_Not subscribed yet — purchase Pro above, then click \"Check Pro status\"._")

                check_pro_btn.click(
                    fn=check_pro_status,
                    inputs=[app_user_id_state],
                    outputs=[pro_group, locked_group],
                    api_visibility="private",
                )
                run_concurrent_btn.click(
                    fn=handle_concurrent_triage,
                    inputs=[app_user_id_state],
                    outputs=[concurrent_output],
                    api_visibility="private",
                )

        gr.Markdown(
            "---\n"
            "*Powered by **NVIDIA Full-Stack AI**: "
            "[Nemotron-Mini-4B](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) (LLM) + "
            "[llama-nemotron-embed-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) (Embeddings, 1024 dims)*\n\n"
            "**+20-36% better retrieval** with NVIDIA embeddings | **16× longer context** (8K tokens) | ChromaDB RAG | Built for GTC 2026\n\n"
            "**Local GPU inference** on T4/A10. Set `HF_TOKEN` for gated models. Set `MODEL_ID` to override LLM."
        )

        demo.load(
            fn=new_app_user_id,
            outputs=[app_user_id_state],
            api_visibility="private",
        ).then(
            fn=render_purchase_widget,
            inputs=[app_user_id_state],
            outputs=[purchase_widget],
            api_visibility="private",
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        mcp_server=True,
        theme=gr.themes.Base(primary_hue="green", neutral_hue="slate"),
    )
