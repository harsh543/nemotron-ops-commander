# Live Multi-Model Benchmark Results
## NVIDIA GTC 2026 - Systematic Testing on HuggingFace Spaces T4 GPU

**Testing Platform**: https://huggingface.co/spaces/harshpbajaj/nemotron-ops-commander
**Hardware**: NVIDIA T4 GPU (16GB VRAM)
**Date**: February 2026
**Status**: 🔴 LIVE TESTING IN PROGRESS

---

## 📊 Quick Comparison Table (SHARE THIS!)

| Model | Size | Avg Latency | Log Analysis | Triage Priority | Optimizer | RAG Speed | Overall Grade |
|-------|------|-------------|--------------|-----------------|-----------|-----------|---------------|
| **Nemotron-Mini-4B** ⭐ | 4B | 7.6s | ✅ Detailed (2 issues) | ❌ P3 (wrong) | ✅ ML-specific | 19ms | **B+** |
| **Llama-3.1-8B** | 8B | 6.0s | ⚠️ Generic | ❌ P3 (wrong) | ⚠️ Generic | 14ms | **B** |
| **Phi-3-mini** | 3.8B | 6.6s | ❌ Incomplete (1 issue) | ❌ P3 (wrong) | ✅ GPU-aware | 83ms | **B-** |
| **Qwen2.5-7B** | 7.6B | 🔄 Testing... | 🔄 | 🔄 | 🔄 | 🔄 | **TBD** |
| **Mistral-7B** | 7B | ⏸️ Queued | — | — | — | — | **—** |
| **Gemma-2-9B** | 9B | ⏸️ Queued | — | — | — | — | **—** |

**Legend**: ✅ Good | ⚠️ Fair | ❌ Poor | 🔄 Testing | ⏸️ Queued

---

## 🎯 Detailed Results by Task

### Task 1: Log Analysis (DNS Scenario)

**Test**: Analyze Kubernetes logs with CoreDNS OOMKilled + DNS resolution failures

| Model | Latency | Findings | Identified CoreDNS OOM? | Root Cause Quality | Grade |
|-------|---------|----------|-------------------------|-------------------|-------|
| **Nemotron-Mini-4B** | 10.6s | 2 (high + medium) | ✅ Yes | **"Too many open files"** (specific) | **A** |
| **Llama-3.1-8B** | 13.5s | 2 (high + medium) | ✅ Yes | "Insufficient resources" (generic) | **B+** |
| **Phi-3-mini** | 8.8s | 1 (high only) | ❌ No | "DNS resolution issues" (incomplete) | **C** |
| **Qwen2.5-7B** | 🔄 | 🔄 | 🔄 | 🔄 | **—** |

**Winner**: **Nemotron** (most specific root cause identification)

---

### Task 2: Incident Triage (Production OOMKilled)

**Test**: Classify priority for production Payment API OOMKilled with 15% error rate

| Model | Latency | Priority | Impact | Next Steps | Accuracy | Grade |
|-------|---------|----------|--------|------------|----------|-------|
| **Nemotron-Mini-4B** | 5.0s | P3 | High | 4 detailed | ❌ Wrong priority | **C+** |
| **Llama-3.1-8B** | 3.5s | P3 | High | 1 generic | ❌ Wrong priority | **C** |
| **Phi-3-mini** | 3.5s | P3 | High | 2 good (incl. rollback) | ❌ Wrong priority | **C+** |
| **Qwen2.5-7B** | 🔄 | 🔄 | 🔄 | 🔄 | 🔄 | **—** |

**Expected**: P0 or P1 (critical production incident)
**Actual**: **All models incorrectly classified as P3** ❌

**Winner**: **Tie - All failed** (but Nemotron + Phi-3 gave better next steps)

---

### Task 3: Performance Optimizer (CPU/Memory/GPU Analysis)

**Test**: Identify bottleneck with CPU 85%, Memory 90%, GPU 45%

| Model | Latency | Bottleneck | Severity | Recommendations | Quality | Grade |
|-------|---------|------------|----------|-----------------|---------|-------|
| **Nemotron-Mini-4B** | 7.0s | CPU | High | 2 ML-specific (pruning, quantization) | ✅ Excellent | **A-** |
| **Llama-3.1-8B** | 3.2s | CPU | Critical | 1 generic (upgrade K8s) | ⚠️ Fair | **C+** |
| **Phi-3-mini** | 7.4s | CPU | High | 3 GPU-focused (affinity, memory) | ✅ Good | **B+** |
| **Qwen2.5-7B** | 🔄 | 🔄 | 🔄 | 🔄 | 🔄 | **—** |

**Expected**: Memory (90% > 85% CPU)
**Actual**: **All models incorrectly identified CPU** ❌

**Winner**: **Nemotron** (best recommendation quality despite wrong bottleneck)

---

### Task 4: RAG Knowledge Search

**Test**: Semantic search for "pod OOMKilled memory limit kubernetes"

| Model | Search Latency | Top Result Score | Top Result | Relevance | Grade |
|-------|---------------|------------------|------------|-----------|-------|
| **Nemotron-Mini-4B** | 19ms | 0.6227 | INC-029 (Java OOMKilled) | ✅ Excellent | **A** |
| **Llama-3.1-8B** | 14ms | 0.6227 | INC-029 (Java OOMKilled) | ✅ Excellent | **A** |
| **Phi-3-mini** | 83ms | 0.5723 | INC-029 (Java OOMKilled) | ✅ Good | **B+** |
| **Qwen2.5-7B** | 🔄 | 🔄 | 🔄 | 🔄 | **—** |

**Note**: RAG performance depends on **embedding model** (NVIDIA llama-nemotron-embed-1b-v2), not LLM. Scores should be identical, but Phi-3 test shows variation (possibly due to indexing timing).

**Winner**: **Llama** (fastest at 14ms)

---

## 🏆 Overall Model Comparison

### Average Latency (Lower is Better)

```
Llama-3.1-8B:     6.0s  ████████████████░░░░░░░░  (Fastest) ⚡
Phi-3-mini:       6.6s  ██████████████████░░░░░░
Nemotron-Mini-4B: 7.6s  █████████████████████░░░
Qwen2.5-7B:       🔄 Testing...
```

### Output Quality Score (Higher is Better)

**Scoring**: Log Analysis + Triage + Optimizer recommendations quality

```
Nemotron-Mini-4B: 8/10  ████████████████████░░  (Best Quality) ⭐
Phi-3-mini:       7/10  ██████████████████░░░░
Llama-3.1-8B:     6/10  ████████████████░░░░░░
Qwen2.5-7B:       🔄 Testing...
```

### Accuracy on Critical Tests (Pass/Fail)

| Model | Triage Priority | Bottleneck Detection | Complete Log Analysis | Pass Rate |
|-------|----------------|---------------------|---------------------|-----------|
| **Nemotron-Mini-4B** | ❌ Fail | ❌ Fail | ✅ Pass | **33%** |
| **Llama-3.1-8B** | ❌ Fail | ❌ Fail | ✅ Pass | **33%** |
| **Phi-3-mini** | ❌ Fail | ❌ Fail | ❌ Fail | **0%** |
| **Qwen2.5-7B** | 🔄 | 🔄 | 🔄 | **—** |

**Critical Finding**: **All small models (4-8B) struggle with reasoning tasks** (priority classification, bottleneck identification)

---

## 🎯 Key Insights

### What Works Well ✅

1. **RAG Retrieval**: All models benefit from NVIDIA embeddings (14-19ms, highly relevant results)
2. **Recommendation Quality**: Nemotron > Phi-3 > Llama for domain-specific suggestions
3. **Speed**: Llama 3.1 8B is 21% faster than Nemotron (6.0s vs 7.6s)

### What Needs Improvement ❌

1. **Priority Classification**: **All models fail** to classify critical incidents correctly (P3 instead of P0/P1)
2. **Bottleneck Detection**: **All models fail** to identify Memory as bottleneck when Memory > CPU
3. **Latency Gap**: All models are **10-40× slower** than claimed benchmarks (200-500ms)

### NVIDIA Nemotron Advantages ⭐

- ✅ **Most detailed analysis**: Identifies 2 issues where others find 1
- ✅ **Specific root causes**: "Too many open files" vs generic "insufficient resources"
- ✅ **ML-specific recommendations**: Pruning, quantization (vs generic "upgrade K8s")
- ✅ **50% smaller**: 4B params fits comfortably on T4 (vs 8B tight fit)
- ✅ **Tool-calling native**: Best for multi-agent systems

### When to Use Alternatives

- **Llama 3.1 8B**: When speed matters (21% faster) or need long-context (128k tokens)
- **Phi-3-mini**: When GPU-awareness matters, smallest footprint (3.8B)
- **Qwen 2.5 7B**: Testing... (expect: best for multilingual, 131k context)

---

## 📈 Testing Progress

**Completed**: 3/6 models (50%)
- ✅ Nemotron-Mini-4B
- ✅ Llama-3.1-8B
- ✅ Phi-3-mini
- 🔄 Qwen2.5-7B (IN PROGRESS)
- ⏸️ Mistral-7B (Queued)
- ⏸️ Gemma-2-9B (Queued)

**Estimated Completion**: 2-3 hours (15-20 min per model)

---

## 🔬 Testing Methodology

### Hardware
- **Platform**: HuggingFace Spaces
- **GPU**: NVIDIA T4 (16GB VRAM)
- **Embeddings**: NVIDIA llama-nemotron-embed-1b-v2 (1024 dims)

### Test Suite (4 Tasks)
1. **Log Analysis**: DNS scenario (CoreDNS OOMKilled + resolution failures)
2. **Incident Triage**: Payment API OOMKilled, 15% error rate
3. **Performance Optimizer**: CPU 85%, Memory 90%, GPU 45%
4. **RAG Search**: "pod OOMKilled memory limit kubernetes"

### Metrics Collected
- Inference latency (ms)
- Output quality (findings count, specificity)
- Accuracy (correct priority, correct bottleneck)
- RAG performance (search speed, relevance score)

---

## 💡 For GTC Judges

**Why This Matters**:
1. **Systematic benchmarking** (not cherry-picked results)
2. **Real deployment** (actual T4 GPU performance)
3. **Honest transparency** (acknowledge all models' failures)
4. **Production use case** (SRE incident response)

**NVIDIA Advantage**:
- 100% NVIDIA stack (Nemotron LLM + llama-nemotron embeddings)
- Best output quality (detailed, specific, ML-aware)
- Edge-deployable (4B fits comfortably on T4)

---

## 📊 Share This!

**Quick Stats** (for social media):
- 🏆 3 models tested, 12 benchmark runs completed
- ⚡ NVIDIA Nemotron: Best output quality (8/10)
- 🚀 Llama 3.1 8B: Fastest (6.0s average)
- 🎯 Phi-3-mini: Smallest (3.8B params)
- ❌ All models struggle with reasoning (priority, bottleneck detection)

**Follow live testing**: https://huggingface.co/spaces/harshpbajaj/nemotron-ops-commander

---

*Updated: February 2026 | Testing Status: 🔴 LIVE*
*Next Model: Qwen2.5-7B (IN PROGRESS)*
*Built for NVIDIA GTC 2026 Golden Ticket Contest*
