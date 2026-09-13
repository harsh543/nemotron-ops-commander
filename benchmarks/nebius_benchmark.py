"""
Nebius Token Factory benchmark — Applied AI / Nebius track evidence.

Runs the model against real incident data (data/sample_incidents/), the
same inputs a user would paste into the app, and measures:

  - quality  : does the triage mention the incident's actual root cause?
  - time     : per-request latency, and wall-clock time for N incidents
               triaged concurrently vs sequentially (the Space's free T4
               can only serialize requests; Token Factory does not)
  - cost     : $ per request, from Nebius' published per-token pricing

It also runs one deliberate failure case (an empty incident) to show the
client fails loudly and predictably rather than silently.

Usage:
    export NEBIUS_API_KEY=...
    python benchmarks/nebius_benchmark.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "huggingface_space"))

SYSTEM_PROMPT = (
    "You are an SRE incident triage assistant. Given an incident summary, "
    "state the most likely root cause in one sentence."
)


def load_incidents(n: int = 5) -> List[Dict]:
    incidents = []
    for path in sorted((ROOT / "data" / "sample_incidents").glob("*.json"))[:n]:
        incidents.append(json.loads(path.read_text()))
    return incidents


def run_quality_and_cost(client, incidents: List[Dict]) -> List[Dict]:
    """Sequential pass: per-incident latency, cost, and a keyword-match quality check."""
    results = []
    for inc in incidents:
        prompt = f"Incident: {inc['summary']}\nService: {inc['service']}\nSeverity: {inc['severity']}"
        text, latency_ms = client.generate(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=200)

        root_cause_words = {w.lower().strip(".,") for w in inc["root_cause"].split()}
        response_words = {w.lower().strip(".,") for w in text.split()}
        overlap = root_cause_words & response_words
        quality_hit = len(overlap) >= 2  # crude but honest: >=2 shared keywords

        usage = client.last_usage or {}
        results.append(
            {
                "id": inc["id"],
                "latency_ms": round(latency_ms, 1),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cost_usd": usage.get("cost_usd"),
                "expected_root_cause": inc["root_cause"],
                "response": text,
                "quality_keyword_match": quality_hit,
            }
        )
    return results


async def run_concurrency_comparison(client, incidents: List[Dict]) -> Dict:
    """Same N incidents, sequential vs concurrent dispatch (asyncio.gather + to_thread)."""
    prompts = [
        f"Incident: {inc['summary']}\nService: {inc['service']}\nSeverity: {inc['severity']}"
        for inc in incidents
    ]

    start = time.time()
    for p in prompts:
        client.generate(p, system_prompt=SYSTEM_PROMPT, max_tokens=150)
    sequential_s = time.time() - start

    start = time.time()
    await asyncio.gather(
        *[
            asyncio.to_thread(client.generate, p, SYSTEM_PROMPT, 150)
            for p in prompts
        ]
    )
    concurrent_s = time.time() - start

    return {
        "n_incidents": len(incidents),
        "sequential_s": round(sequential_s, 2),
        "concurrent_s": round(concurrent_s, 2),
        "speedup": round(sequential_s / concurrent_s, 2) if concurrent_s > 0 else None,
    }


def run_failing_case(client) -> Dict:
    """Deliberately malformed input — must fail predictably, not hang or silently return junk."""
    try:
        client.generate("", system_prompt=SYSTEM_PROMPT, max_tokens=0)
        return {"failed_as_expected": False, "note": "empty/zero-token request unexpectedly succeeded"}
    except Exception as exc:
        return {"failed_as_expected": True, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    from inference import get_client  # huggingface_space/inference.py

    client = get_client()
    if client.backend != "nebius":
        print(
            "WARNING: NEBIUS_API_KEY not set (or openai package missing) — "
            f"running against '{client.backend}' backend instead of Nebius Token Factory.\n"
        )

    print(f"Backend: {client.get_active_model()}\n")

    incidents = load_incidents(5)

    print("=== Quality / latency / cost (sequential, 5 real incidents) ===")
    results = run_quality_and_cost(client, incidents)
    have_cost = any(r["cost_usd"] is not None for r in results)
    for r in results:
        cost_str = f"${r['cost_usd']}" if r["cost_usd"] is not None else f"{r['prompt_tokens']}+{r['completion_tokens']}tok"
        print(
            f"  {r['id']}: {r['latency_ms']}ms  "
            f"{cost_str}  "
            f"quality_match={r['quality_keyword_match']}"
        )
    hit_rate = statistics.mean(1 if r["quality_keyword_match"] else 0 for r in results)
    mean_latency = statistics.mean(r["latency_ms"] for r in results)
    print(f"\n  Keyword-match quality rate: {hit_rate:.0%}")
    print(f"  Mean latency: {mean_latency:.0f}ms")
    if have_cost:
        total_cost = sum(r["cost_usd"] or 0 for r in results)
        print(f"  Total cost for {len(results)} requests: ${total_cost:.6f}")
    else:
        total_tokens = sum((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in results)
        print(
            f"  Total tokens for {len(results)} requests: {total_tokens} "
            "(set NEBIUS_PROMPT_COST_PER_TOKEN / NEBIUS_COMPLETION_COST_PER_TOKEN "
            "from your Nebius billing dashboard to see $ cost)"
        )

    print("\n=== Concurrency: same 5 incidents, sequential vs concurrent ===")
    concurrency = asyncio.run(run_concurrency_comparison(client, incidents))
    print(f"  Sequential: {concurrency['sequential_s']}s")
    print(f"  Concurrent: {concurrency['concurrent_s']}s")
    print(f"  Speedup: {concurrency['speedup']}x")

    print("\n=== Deliberate failing case (empty incident, max_tokens=0) ===")
    failure = run_failing_case(client)
    print(f"  {failure}")

    out_path = ROOT / "benchmarks" / "nebius_benchmark_results.json"
    out_path.write_text(
        json.dumps(
            {
                "backend": client.get_active_model(),
                "quality_cost_latency": results,
                "concurrency": concurrency,
                "failing_case": failure,
            },
            indent=2,
        )
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
