"""Streamlit dashboard for NemOps agent."""

import streamlit as st
import json
from nemops.agent import run_agent, TOOLS
from nemops.tools.gpu_health import gpu_health_check
from nemops.llm import NemotronClient

st.set_page_config(page_title="NemOps", page_icon="🔧", layout="wide")

st.title("🔧 NemOps — GPU Infrastructure Monitor")
st.caption("Powered by NVIDIA Nemotron 3 Nano via Ollama")

# Sidebar — system status
with st.sidebar:
    st.header("System Status")

    client = NemotronClient()
    model_ok = client.health_check()

    st.metric("Nemotron Model", "✅ Online" if model_ok else "❌ Offline")
    st.metric("Model", client.model)
    st.metric("GPU Mode", "Mock (Demo)")

    if not model_ok:
        st.error(f"Start Ollama and pull model:\n`ollama pull {client.model}`")

    st.divider()
    st.header("Quick Actions")
    if st.button("🔍 Health Check All GPUs"):
        st.session_state["quick_query"] = (
            "Check the health of all GPUs and report any issues."
        )
    if st.button("🔥 Simulate ECC Failure"):
        st.session_state["quick_query"] = (
            "GPU 0 is showing rising ECC errors and XID 63 events. "
            "Investigate the failure pattern, search incidents, and generate an alert."
        )
    if st.button("🌡️ Thermal Investigation"):
        st.session_state["quick_query"] = (
            "Multiple GPUs are thermal throttling. Temperature is 89°C with fans at 100%. "
            "Diagnose the root cause and recommend remediation."
        )

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Agent Query")
    default_query = st.session_state.pop("quick_query", "")
    query = st.text_area(
        "Ask NemOps about your GPU infrastructure:",
        value=default_query,
        height=100,
        placeholder="e.g., Check GPU health and investigate any anomalies...",
    )

    if st.button("🚀 Run Agent", type="primary", disabled=not model_ok):
        if query:
            with st.spinner("Nemotron is analyzing your GPU infrastructure..."):
                result = run_agent(query, verbose=False)

            if "error" in result and not result.get("offline_demo"):
                st.error(result["error"])

            if "response" in result:
                st.success(f"Completed in {result['steps']} steps")
                st.markdown(result["response"])

                # Show tool calls
                if result.get("tool_calls"):
                    with st.expander(
                        f"🔧 Tool Calls ({len(result['tool_calls'])})", expanded=False
                    ):
                        for tc in result["tool_calls"]:
                            step = tc.get("step", "-")
                            args = tc.get("args", {})
                            st.code(
                                f"Step {step}: {tc.get('tool', 'tool')}({json.dumps(args)})",
                                language="python",
                            )

with col2:
    st.header("Live GPU Status")

    # Show current mock GPU health
    health = gpu_health_check()
    if "gpus" in health:
        summary = health["summary"]
        status_color = {
            "healthy": "🟢",
            "warning": "🟡",
            "critical": "🔴",
        }

        st.metric(
            "Node Status",
            f"{status_color.get(summary['overall_status'], '⚪')} {summary['overall_status'].upper()}",
        )

        cols = st.columns(3)
        cols[0].metric("Healthy", summary["healthy"])
        cols[1].metric("Warning", summary["warnings"])
        cols[2].metric("Critical", summary["critical"])

        for gpu in health["gpus"]:
            icon = status_color.get(gpu["status"], "⚪")
            with st.expander(f"{icon} GPU {gpu['gpu_id']}: {gpu['name'][:30]}"):
                st.metric("Temperature", f"{gpu['temperature_c']:.0f}°C")
                st.metric("Utilization", f"{gpu['utilization_pct']:.0f}%")
                st.metric("Memory", f"{gpu['memory_used_gb']:.1f}/{gpu['memory_total_gb']:.0f} GB")
                st.metric("Power", f"{gpu['power_draw_w']:.0f}W / {gpu['power_limit_w']:.0f}W")
                if gpu["issues"]:
                    for issue in gpu["issues"]:
                        st.warning(issue)
