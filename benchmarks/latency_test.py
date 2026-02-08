"""Latency benchmark for Nemotron inference."""

from __future__ import annotations

import asyncio
import time

from models.nemotron_client import NemotronClient, NemotronConfig


async def run_latency_test(prompt: str, iterations: int = 5) -> float:
    """Run latency test and return average latency in ms."""

    client = NemotronClient(NemotronConfig())
    latencies = []
    for _ in range(iterations):
        start = time.time()
        await client.generate(prompt)
        latencies.append((time.time() - start) * 1000)
    return sum(latencies) / len(latencies)


if __name__ == "__main__":
    avg = asyncio.run(run_latency_test("Hello"))
    print(f"Average latency: {avg:.2f} ms")
