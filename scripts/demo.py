"""End-to-end demo script."""

from __future__ import annotations

import asyncio

from agents.log_analyzer import LogAnalyzerAgent
from models.nemotron_client import NemotronClient, NemotronConfig


async def main() -> None:
    client = NemotronClient(NemotronConfig())
    agent = LogAnalyzerAgent(client)

    payload = {
        "logs": [
            {
                "timestamp": "2026-02-07T10:00:00Z",
                "source": "kubelet",
                "message": "Readiness probe failed: connection refused",
                "labels": {"pod": "payments-api"},
            }
        ],
        "system": "payments",
        "environment": "prod",
    }

    result = await agent.run(payload)
    print(result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
