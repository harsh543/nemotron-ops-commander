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

**AI-Powered Incident Response System** using NVIDIA Nemotron with RAG over 30 real-world incidents.

## What This Demo Does

| Tab | Description |
|-----|-------------|
| **Log Analysis** | Paste K8s/application logs for AI root-cause analysis |
| **Incident Triage** | Describe an incident for severity classification (P0-P4) |
| **Performance Optimizer** | Input system metrics for optimization recommendations |
| **Knowledge Search (RAG)** | Semantic search over 30 real-world SRE incidents |

## How It Works

- **LLM Inference**: HuggingFace Inference API (serverless, CPU-safe)
- **RAG Pipeline**: ChromaDB (embedded) + sentence-transformers embeddings
- **Knowledge Base**: 30 real-world incidents from K8s, AWS, Azure, StackOverflow, GitHub
- **Multi-Agent Architecture**: 4 specialized SRE agents (Log Analyzer, Incident Triager, Remediation Suggester, Performance Optimizer)

## Try It

1. Click any tab
2. Use the pre-filled sample data or enter your own
3. Click the action button
4. See AI-powered analysis results

> **Note**: This runs on CPU-only free tier. First request may take 10-30s as models warm up.

## Architecture

```
User Query -> Gradio UI -> Agent (prompt builder) -> HF Inference API -> JSON Parser -> Formatted Output
                                                  -> ChromaDB RAG (for knowledge search)
```

## Configuration

Set these as **Space Secrets** for better performance:

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | HuggingFace token (for higher Inference API rate limits) |
| `MODEL_ID` | Override default model (e.g., `nvidia/Nemotron-Mini-4B-Instruct`) |

---

*Built for the NVIDIA GTC Golden Ticket Contest*
*Powered by [NVIDIA Nemotron](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) | ChromaDB RAG | sentence-transformers*
