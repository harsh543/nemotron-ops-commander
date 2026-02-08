"""Throughput benchmark for Nemotron inference."""

from __future__ import annotations

import asyncio
import time

from models.nemotron_client import NemotronClient, NemotronConfig


async def run_throughput_test(prompt: str, requests: int = 10) -> float:
    """Run throughput test and return requests per second."""

    client = NemotronClient(NemotronConfig())
    start = time.time()
    await asyncio.gather(*[client.generate(prompt) for _ in range(requests)])
    elapsed = time.time() - start
    return requests / elapsed


if __name__ == "__main__":
    rps = asyncio.run(run_throughput_test("Hello"))
    print(f"Throughput: {rps:.2f} req/s")
