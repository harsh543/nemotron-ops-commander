"""
Latency benchmark for Nemotron inference.

Compares SGLang-optimized vs standard Transformers pipeline.

Usage:
    python benchmarks/latency_test.py
    python benchmarks/latency_test.py --iterations 20 --warmup 5
    python benchmarks/latency_test.py --output benchmarks/results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_PROMPTS = [
    "Analyze this log: ERROR OOMKilled Container payment-api exceeded memory limit 2Gi",
    "Triage this incident: API latency spiked to 5 seconds, error rate 15%, 3 pods restarting",
    "Suggest remediation: PostgreSQL connection pool exhausted, 800 active connections, max is 200",
    "Analyze: CrashLoopBackOff on pod user-service, ImagePullBackOff from private registry",
    "Optimize: CPU at 85%, memory at 92%, GC pauses 2.3 seconds on JVM service",
]


async def run_benchmark(
    iterations: int = 10,
    warmup: int = 3,
    use_sglang: bool = True,
) -> Dict:
    """Run latency benchmark and return statistics."""
    from models.nemotron_client import NemotronClient, NemotronConfig

    backend = "sglang" if use_sglang else "transformers"
    print(f"\n  Backend: {backend} | Iterations: {iterations} | Warmup: {warmup}")

    config = NemotronConfig(use_sglang=use_sglang)
    try:
        client = NemotronClient(config)
    except Exception as e:
        print(f"  ERROR: Could not init model: {e}")
        return {"backend": backend, "error": str(e)}

    # Warmup
    print(f"  Warming up ({warmup} requests)...", end="", flush=True)
    for i in range(warmup):
        await client.generate("Hello, warmup request.")
        print(".", end="", flush=True)
    print(" done")

    # Benchmark
    latencies: List[float] = []
    print(f"  Running {iterations} iterations...", end="", flush=True)
    for i in range(iterations):
        prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        start = time.perf_counter()
        resp = await client.generate(
            prompt=prompt,
            system_prompt="You are an SRE assistant. Respond concisely.",
        )
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        print(".", end="", flush=True)
    print(" done")

    # Compute stats
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    stats = {
        "backend": backend,
        "model": config.model_name,
        "iterations": iterations,
        "mean_ms": round(statistics.mean(latencies), 1),
        "median_ms": round(statistics.median(latencies), 1),
        "std_ms": round(statistics.stdev(latencies), 1) if n > 1 else 0,
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "p50_ms": round(latencies_sorted[int(n * 0.5)], 1),
        "p90_ms": round(latencies_sorted[int(n * 0.9)], 1),
        "p95_ms": round(latencies_sorted[int(n * 0.95)], 1),
        "p99_ms": round(latencies_sorted[min(int(n * 0.99), n - 1)], 1),
        "throughput_rps": round(1000 / statistics.mean(latencies), 2),
    }
    return stats


def print_results(results: List[Dict]) -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 65}")
    print(f"  {'Metric':<25} ", end="")
    for r in results:
        print(f"{'  ' + r.get('backend', '?'):<20}", end="")
    print()
    print(f"{'=' * 65}")

    metrics = [
        ("Mean (ms)", "mean_ms"),
        ("Median (ms)", "median_ms"),
        ("Std Dev (ms)", "std_ms"),
        ("Min (ms)", "min_ms"),
        ("Max (ms)", "max_ms"),
        ("P50 (ms)", "p50_ms"),
        ("P90 (ms)", "p90_ms"),
        ("P95 (ms)", "p95_ms"),
        ("P99 (ms)", "p99_ms"),
        ("Throughput (req/s)", "throughput_rps"),
    ]

    for label, key in metrics:
        print(f"  {label:<25} ", end="")
        for r in results:
            val = r.get(key, "N/A")
            print(f"  {str(val):<18}", end="")
        print()

    # Speedup
    if len(results) == 2 and all("mean_ms" in r for r in results):
        slower = max(r["mean_ms"] for r in results)
        faster = min(r["mean_ms"] for r in results)
        if faster > 0:
            print(f"\n  Speedup: {slower/faster:.1f}x")

    print(f"{'=' * 65}")


async def amain(args: argparse.Namespace) -> None:
    print(f"\n{'=' * 65}")
    print(f"  Nemotron-Ops-Commander Latency Benchmark")
    print(f"{'=' * 65}")

    results = []

    if args.compare:
        # Run both backends
        print("\n  [1/2] SGLang backend:")
        r1 = await run_benchmark(args.iterations, args.warmup, use_sglang=True)
        results.append(r1)

        print("\n  [2/2] Standard Transformers backend:")
        r2 = await run_benchmark(args.iterations, args.warmup, use_sglang=False)
        results.append(r2)
    else:
        r = await run_benchmark(args.iterations, args.warmup, use_sglang=args.sglang)
        results.append(r)

    print_results(results)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\n  Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nemotron latency benchmark")
    parser.add_argument("--iterations", type=int, default=10, help="Number of test iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--sglang", action="store_true", default=True, help="Use SGLang backend")
    parser.add_argument("--compare", action="store_true", help="Compare SGLang vs Transformers")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
