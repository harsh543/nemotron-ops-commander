"""Base agent implementation with observability and error handling."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

import structlog
from pydantic import BaseModel, ValidationError

from models.nemotron_client import NemotronClient

logger = structlog.get_logger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class AgentError(RuntimeError):
    """Raised when an agent fails to complete a request."""


class BaseAgent(ABC):
    """Base class for all agents."""

    name: str = "base-agent"
    system_prompt: str = ""

    def __init__(self, client: NemotronClient) -> None:
        self.client = client

    @abstractmethod
    async def run(self, payload: Dict[str, Any]) -> BaseModel:
        """Run the agent and return a structured result."""

    async def _generate_text(self, prompt: str) -> tuple[str, float]:
        """Generate text from Nemotron with timing and logging."""

        start = time.time()
        try:
            response = await self.client.generate(prompt=prompt, system_prompt=self.system_prompt)
            logger.info(
                "agent.generate.success",
                agent=self.name,
                latency_ms=response.latency_ms,
            )
            return response.text, response.latency_ms
        except Exception as exc:  # noqa: BLE001
            logger.error("agent.generate.error", agent=self.name, error=str(exc))
            raise AgentError(str(exc)) from exc
        finally:
            elapsed = (time.time() - start) * 1000
            logger.debug("agent.generate.complete", agent=self.name, elapsed_ms=elapsed)

    def _parse_json(self, text: str, schema: Type[SchemaT]) -> SchemaT:
        """Parse JSON from model output into the requested schema."""

        try:
            return schema.model_validate_json(text)
        except ValidationError:
            try:
                parsed = json.loads(text)
                return schema.model_validate(parsed)
            except Exception as exc:  # noqa: BLE001
                logger.error("agent.parse.error", agent=self.name, error=str(exc))
                raise AgentError("Failed to parse model output") from exc
