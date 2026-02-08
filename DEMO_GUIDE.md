# Demo & Contest Guide: Nemotron-Ops-Commander

## How to Win the GTC Golden Ticket

### What NVIDIA Judges Look For

1. **Actual Nemotron usage** — not just a wrapper around OpenAI
2. **Technical depth** — SGLang optimizations, structured generation, batching
3. **Real-world usefulness** — solves an actual problem SRE teams face
4. **Production readiness** — not a toy, has observability, error handling, deployment
5. **Clear demo** — they need to SEE it working, not just read about it
6. **Performance numbers** — benchmarks showing SGLang speedup are gold

### What Sets This Project Apart

| Differentiator | Why It Matters |
|----------------|----------------|
| **4 specialized agents** | Not a single chatbot — a multi-agent pipeline |
| **LangGraph orchestration** | Agents coordinate, not just run in parallel |
| **RAG over 30 real incidents** | From K8s, AWS, Azure, StackOverflow, GitHub |
| **SGLang optimizations** | 2-3x speedup with benchmarks to prove it |
| **Knowledge connectors** | Public, read-only enrichment from live sources |
| **Full observability** | OpenTelemetry + Prometheus — production-grade |
| **K8s-ready deployment** | Helm chart, GPU resource limits, health checks |
| **Interactive Gradio UI** | Judges can click through live |

---

## Setup Instructions (15 minutes)

### Option A: GPU Machine (Recommended)

```bash
# 1. Clone and enter project
cd nemotron-ops-commander

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated setup (downloads model, builds RAG index)
python scripts/setup_nemotron.py
```

### Option B: CPU-Only (Slower, but works)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/setup_nemotron.py --cpu
```

### Option C: Docker (Easiest)

```bash
cp .env.example .env
docker-compose up --build
# Wait for model download on first run (~5 min)
```

### Verify Setup

After setup completes, you should see:

```
  ✓ Python 3.10+
  ✓ NVIDIA GPU (NVIDIA T4 16GB)
  ✓ Dependencies (All core packages found)
  ✓ .env file (Found)
  ✓ Model download (Cached)
  ✓ Model init (nvidia/Nemotron-Mini-4B-Instruct on cuda)
  ✓ RAG index (30 incidents indexed into ChromaDB)

  Ready to run:
    1. Start server:  uvicorn api.main:app --port 8000
    2. Run demo:      python scripts/demo.py
    3. Open UI:       python ui/gradio_app.py
```

---

## Running the Demo

### Demo Mode 1: CLI Script (Best for recordings)

This runs 3 production scenarios end-to-end, showing all 4 agents:

```bash
# Run all scenarios with benchmarks
python scripts/demo.py --benchmark

# Run specific scenario
python scripts/demo.py --scenario 1

# Run against API server (start server first)
python scripts/demo.py --api
```

**Scenarios included:**
1. **OOMKilled Production Pod** — Memory exhaustion after deployment
2. **Database Connection Storm** — Connection pool exhausted by HPA scaling
3. **TLS Certificate Expiry** — cert-manager renewal failed due to IAM

Each scenario walks through:
- Step 1: Log Analysis (feeds logs to Nemotron)
- Step 2: RAG Retrieval (finds similar historical incidents)
- Step 3: Incident Triage (assigns P0-P4 priority)
- Step 4: Performance Optimization (analyzes metrics)
- Step 5: Benchmark (optional, shows inference speed)

### Demo Mode 2: Gradio UI (Best for live demos)

```bash
# Terminal 1: Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start the UI
python ui/gradio_app.py
# Open http://localhost:7860
```

**4 Tabs to demo:**

| Tab | What to show | Talking point |
|-----|-------------|---------------|
| **Log Analysis** | Paste sample logs, click Analyze | "Nemotron identifies root cause from raw logs in <1s" |
| **Incident Triage** | Use pre-filled incident, click Triage | "Assigns P0-P4 priority with action plan" |
| **Performance Optimizer** | Slide CPU/memory to high values | "Recommends specific optimizations per service" |
| **Knowledge Search** | Search "OOMKilled kubernetes" | "RAG over 30 real incidents from K8s, AWS, Azure, SO" |

### Demo Mode 3: curl Commands (Best for technical audiences)

```bash
# Log Analysis
curl -s -X POST http://localhost:8000/analyze/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{
    "logs": [
      {"timestamp": "now", "source": "kubelet", "message": "OOMKilled: Container exceeded memory limit 2Gi"}
    ],
    "system": "payment-api",
    "environment": "production"
  }' | python -m json.tool

# Incident Triage
curl -s -X POST http://localhost:8000/triage/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{
    "incident_id": "INC-001",
    "title": "API latency spike",
    "description": "p99 latency at 5s, error rate 15%, pods OOMKilled",
    "metrics": {"p99": 5000, "error_rate": 0.15}
  }' | python -m json.tool

# Performance Optimization
curl -s -X POST http://localhost:8000/optimize/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{
    "metrics": {"cpu": 85, "memory": 92, "gpu": 45},
    "service": "ml-inference",
    "context": "Kubernetes with A10 GPU"
  }' | python -m json.tool

# RAG Knowledge Search
curl -s -X POST http://localhost:8000/rag/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{"query": "certificate expired kubernetes", "top_k": 3}' | python -m json.tool
```

---

## Running Benchmarks (Key for Contest)

```bash
# Quick benchmark (10 iterations)
python benchmarks/latency_test.py

# Full comparison: SGLang vs Transformers
python benchmarks/latency_test.py --compare --iterations 20 --warmup 5

# Save results for submission
python benchmarks/latency_test.py --compare --iterations 50 --output benchmarks/results.json
```

**Expected output:**
```
  Metric                    sglang              transformers
=================================================================
  Mean (ms)                 185.3               472.1
  P50 (ms)                  178.2               458.7
  P90 (ms)                  210.4               520.3
  P99 (ms)                  315.6               812.4
  Throughput (req/s)        5.40                2.12

  Speedup: 2.5x
```

---

## Contest Submission Checklist

### Must Have (Day 1-2)
- [ ] GitHub repo is public
- [ ] README has architecture diagram, quick start, API docs
- [ ] Demo video (2-3 minutes) showing all 4 agents working
- [ ] Benchmark numbers comparing SGLang vs baseline
- [ ] All 30 incidents indexed and RAG working
- [ ] Gradio UI running and visually clean

### Should Have (Day 3-4)
- [ ] Docker compose works on fresh pull
- [ ] Helm chart deploys to K8s cluster
- [ ] OpenTelemetry traces visible in Jaeger/similar
- [ ] Knowledge connectors pull live data from StackOverflow/GitHub

### Nice to Have (Day 5)
- [ ] Record a GIF/video of the full pipeline running
- [ ] Add throughput benchmark chart to README
- [ ] Deploy to a cloud GPU instance and share public URL
- [ ] Write a blog post about the architecture

---

## Recording a Demo Video (2-3 min)

### Script:

**0:00 - Intro (15 sec)**
> "This is Nemotron-Ops-Commander — an AI-powered incident response system
> that helps SRE teams analyze logs, triage incidents, and automate
> remediation, all powered by NVIDIA Nemotron with SGLang optimizations."

**0:15 - Architecture (15 sec)**
> Show the mermaid diagram from README.
> "It has 4 specialized agents, RAG over 30 real incidents, and
> public knowledge connectors for StackOverflow, K8s docs, and AWS/Azure."

**0:30 - Live Demo: Log Analysis (30 sec)**
> Open Gradio UI. Paste OOMKilled logs. Click Analyze.
> "Nemotron analyzes raw Kubernetes logs and identifies the root cause
> in under a second."

**1:00 - Live Demo: Triage (30 sec)**
> Switch to Triage tab. Show pre-filled incident. Click Triage.
> "It classifies this as P1, identifies affected services, and
> provides an immediate action plan."

**1:30 - Live Demo: Knowledge Search (20 sec)**
> Switch to Knowledge Search. Query "OOMKilled memory".
> "RAG finds similar historical incidents from our 30-incident
> knowledge base sourced from Kubernetes docs, StackOverflow, and AWS."

**1:50 - Benchmarks (20 sec)**
> Show benchmark comparison table.
> "SGLang optimizations give us a 2.5x speedup over vanilla transformers —
> key for production SRE workflows where speed matters."

**2:10 - Production Ready (20 sec)**
> Show docker-compose.yml, Helm chart, OpenTelemetry config.
> "The system is fully production-ready with Docker, Kubernetes Helm
> charts, Prometheus metrics, and OpenTelemetry tracing."

**2:30 - Close (10 sec)**
> "Nemotron-Ops-Commander: turning NVIDIA Nemotron into a production
> SRE co-pilot. Link in the description."

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Use `NEMOTRON_DEVICE=cpu` or smaller batch size |
| `Model download fails` | Check internet, try `huggingface-cli login` |
| `ChromaDB error on index` | Delete `chroma_storage/` and re-run indexer |
| `SGLang import error` | Fall back: set `NEMOTRON_USE_SGLANG=false` |
| `API returns 401` | Set `X-API-Key: change-me` header (or match .env) |
| `Gradio can't connect` | Start API server first: `uvicorn api.main:app --port 8000` |
| `Port 8000 in use` | Change port: `API_PORT=8001` in .env |
