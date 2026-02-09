# NemOps Architecture

## System Overview

NemOps is an agentic GPU infrastructure monitoring system powered by NVIDIA Nemotron 3 Nano.
It uses a ReAct (Reason + Act) loop to analyze GPU health, diagnose failures using RAG over
historical incidents, and generate structured remediation plans.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interfaces                         │
│                                                             │
│   CLI (nemops)      Streamlit (ui/dashboard.py)    MCP      │
│        │                    │                       │       │
└────────┼────────────────────┼───────────────────────┼───────┘
         │                    │                       │
         ▼                    ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     Agent Layer                              │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  agent.py — ReAct Loop                               │   │
│  │                                                      │   │
│  │  1. User query → Nemotron (tool calling)             │   │
│  │  2. Parse tool_calls from response                   │   │
│  │  3. Execute tool → feed result back                  │   │
│  │  4. Repeat until done or max_steps                   │   │
│  │  5. Return final analysis                            │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  llm.py — NemotronClient (httpx)                     │   │
│  │  POST http://localhost:11434/v1/chat/completions      │   │
│  │  Model: nemotron-3-nano:30b-cloud (16GB Mac)          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Tool Layer                              │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ gpu_health   │  │ diagnostics  │  │  alert_gen       │  │
│  │ _check()     │  │ run_         │  │  generate_       │  │
│  │              │  │ diagnostic() │  │  alert()         │  │
│  │ Mock mode:   │  │ 5 tests:     │  │ OpsGenie-style   │  │
│  │ 7 scenarios  │  │ memory_stress│  │ escalation       │  │
│  │ Real mode:   │  │ compute_     │  │ policies         │  │
│  │ pynvml       │  │ stress, etc. │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ search_incidents() — RAG                             │   │
│  │ ChromaDB + SentenceTransformer (all-MiniLM-L6-v2)    │   │
│  │ 15 real GPU incident patterns                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **User Input** → Query about GPU infrastructure (CLI, Streamlit, or MCP)
2. **Agent Loop** → Nemotron reasons about which tools to call
3. **Tool Execution** → GPU health check, incident search, diagnostics, alerts
4. **RAG Search** → Semantic similarity over 15 historical GPU incidents in ChromaDB
5. **Final Report** → Structured analysis with reasoning chain and remediation steps

## Key Design Decisions

### Why httpx instead of OpenAI client?
- Lighter dependency (httpx vs openai + its deps)
- Direct control over request/response handling
- Ollama's API is OpenAI-compatible but we only need chat completions
- Simpler error handling and timeout management

### Why functional tools instead of classes?
- Tools are stateless — each call is independent
- Simpler testing (just call the function)
- Matches the tool-calling interface (function name → function execution)
- Aligned with MCP's tool pattern

### Why mock GPU data by default?
- Runs on any machine (Mac, Linux, no GPU required)
- 7 weighted scenarios provide realistic variety for demos
- Real mode (pynvml) available for actual GPU servers
- Same interface whether mock or real — agent code doesn't change

### Why ChromaDB for RAG?
- Embedded database — no external service to run
- Persistent storage — survives process restarts
- Cosine similarity with sentence-transformers for semantic search
- Simple Python API, easy to seed and query

## File Structure

```
nemops/
├── pyproject.toml              # Dependencies and entry points
├── setup.sh                    # One-click setup (Ollama + venv + seed)
├── .env.example                # Environment configuration
├── configs/
│   └── agent_config.yaml       # Agent behavior config
├── src/nemops/
│   ├── __init__.py
│   ├── agent.py                # ReAct loop + CLI entry point
│   ├── llm.py                  # NemotronClient (httpx → Ollama)
│   ├── mcp_server.py           # MCP server (stdio)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── gpu_health.py       # GPU health check (mock + real)
│   │   ├── incident_rag.py     # RAG search over ChromaDB
│   │   ├── diagnostics.py      # 5 GPU diagnostic tests
│   │   └── alert_gen.py        # Alert generation + escalation
│   └── data/
│       ├── incidents.json      # 15 GPU incident patterns
│       └── seed_incidents.py   # ChromaDB seeder
├── ui/
│   └── dashboard.py            # Streamlit dashboard
├── tests/
│   ├── test_agent.py
│   ├── test_tools.py
│   └── test_mcp.py
└── docs/
    └── architecture.md         # This file
```
