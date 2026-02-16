---
title: Nemotron-Ops-Commander
emoji: "\U0001F6E1\uFE0F"
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: AI-Powered Incident Response for SRE Teams
---

# Nemotron-Ops-Commander

**AI-Powered Incident Response System** for SRE teams — powered by **100% NVIDIA AI Stack** with RAG over 30 real-world production incidents.

## 🚀 NVIDIA Full-Stack AI

| Component | Technology | Key Benefit |
|-----------|-----------|-------------|
| **LLM** | [Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) | Native tool-calling, 50% smaller than 8B models, T4-compatible |
| **Embeddings** | [llama-nemotron-embed-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) (1024 dims) | **+20-36% better retrieval**, 16× longer context (8K tokens) |
| **GPU** | NVIDIA T4 (16GB VRAM) | Local inference, edge-deployable |
| **Stack** | PyTorch + CUDA + Transformers | Optimized NVIDIA ecosystem |

When a production service goes down, this system automates the first critical minutes: analyzing logs, classifying severity, identifying root causes, and surfacing similar historical incidents with proven resolutions.

> **[GitHub Repository](https://github.com/harshpbajaj/nemotron-ops-commander)** for full source code, architecture docs, and local deployment instructions.

---

## What This Demo Does

| Tab | Agent | What You Get |
|-----|-------|-------------|
| **Log Analysis** | Log Analyzer | Paste K8s/application logs and get structured findings — severity, cited evidence from log lines, confidence scores, root cause hypothesis, and actionable recommendations |
| **Incident Triage** | Incident Triager | Describe an incident and get P0-P4 severity classification, impact assessment, affected service identification, and prioritized next steps |
| **Performance Optimizer** | Performance Optimizer | Input CPU/memory/GPU metrics and get bottleneck identification with targeted optimization actions, rationale, and risk levels |
| **Knowledge Search** | RAG Engine | Semantic search over 30 curated real-world SRE incidents from Kubernetes, AWS, Azure, StackOverflow, and GitHub — returns similar incidents with resolutions in <50ms |

---

## How It Works

### Inference

- **On GPU Spaces (T4/A10)**: Loads the model directly onto the GPU using `transformers` with `torch.float16` and `device_map="auto"` for fast local inference (~200-500ms per query)
- **On CPU Spaces**: Falls back to the HuggingFace Inference API (serverless)
- **Model priority**: Nemotron-Mini-4B-Instruct (primary), Phi-3-mini-4k-instruct (fallback)

### RAG Pipeline (NVIDIA-Powered)

- **ChromaDB** (embedded, runs in `/tmp`) indexes 30 real-world incidents at startup
- **NVIDIA llama-nemotron-embed-1b-v2** generates embeddings for semantic search
  - **1024 dimensions** (vs 384 for previous model)
  - **+20-36% better retrieval accuracy** (vs all-MiniLM-L6-v2)
  - **8,192 token context** (vs 512) — handles long logs without truncation
  - **Matryoshka embeddings** — configurable dimensions for storage/accuracy trade-off
- Incidents cover: OOMKilled pods, CrashLoopBackOff, Node NotReady, database connection storms, TLS expiry, DNS failures, EKS scaling, AKS upgrades, memory leaks, GC pauses, failed deployments, and more

### Multi-Agent Architecture

Four specialized SRE agents, each with domain-specific system prompts and structured JSON output schemas:

```
User Query --> Gradio UI --> Agent (prompt + system prompt) --> LLM --> JSON Parser --> Formatted Output
                                                            --> ChromaDB RAG (knowledge search)
```

---

## Try It

1. Click any tab
2. Use the pre-filled sample data or enter your own scenarios
3. Click the action button
4. See AI-powered analysis with structured results

**Recommended demo order** (start with the fastest tab):
1. **Knowledge Search** — instant semantic search results
2. **Log Analysis** — click "Analyze Logs" with the pre-filled OOMKill scenario
3. **Incident Triage** — click "Triage Incident" for priority classification
4. **Performance Optimizer** — adjust metric sliders, click "Analyze & Optimize"

---

## Knowledge Base — 30 Real-World Incidents

| Source | Count | Topics |
|--------|-------|--------|
| Kubernetes | 8 | OOMKilled, CrashLoopBackOff, Node NotReady, DNS, PVC |
| Database | 4 | Connection storms, deadlocks, replication lag |
| Networking | 3 | TLS cert expiry, DNS resolution, LB timeouts |
| AWS | 4 | EKS scaling, S3 throttling, EC2 limits, IAM |
| Azure | 3 | AKS upgrades, App Gateway 502, VM scaling |
| Application | 4 | Memory leaks, thread pools, GC pauses, async errors |
| CI/CD | 4 | Failed deploys, Helm issues, image pull, rollbacks |

---

## Configuration

Set these as **Space Secrets** for optimal performance:

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | HuggingFace token — required for gated models (Nemotron) and higher Inference API rate limits |
| `MODEL_ID` | Override default model (default: `nvidia/Nemotron-Mini-4B-Instruct`) |

### Hardware Recommendation

- **T4 Small** (16GB VRAM) — recommended for local GPU inference, Nemotron-Mini-4B fits in ~8GB fp16
- **CPU Basic** — works via HF Inference API fallback, but slower (10-30s per query)

---

## Full Project

This Space is a deployment of [Nemotron-Ops-Commander](https://github.com/harshpbajaj/nemotron-ops-commander), which includes:

- **FastAPI backend** with authentication, rate limiting, and REST APIs
- **LangGraph orchestration** for multi-agent pipelines
- **SGLang optimization** for 2.5x inference speedup
- **OpenTelemetry + Prometheus** observability stack
- **Docker + Kubernetes** deployment manifests
- **Knowledge connectors** for StackOverflow, K8s Docs, AWS/Azure Docs, GitHub Issues
- **NemOps Local** — GPU infrastructure monitor using Nemotron 3 Nano + Ollama

---

*Built for the NVIDIA GTC 2026 Golden Ticket Developer Contest | #NVIDIAGTC*

*Powered by **100% NVIDIA AI Stack**:*
- *[NVIDIA Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) (LLM)*
- *[NVIDIA llama-nemotron-embed-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) (Embeddings)*
- *ChromaDB RAG | NVIDIA T4 GPU | PyTorch + CUDA*

*Author: [Harsh Bajaj](https://github.com/harshpbajaj) | [Comprehensive Benchmarks](https://github.com/harshpbajaj/nemotron-ops-commander/blob/main/benchmarks/GTC_2026_Submission_Package.md)*
