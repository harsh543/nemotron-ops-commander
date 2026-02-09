"""Ollama/Nemotron LLM client with native tool calling support."""

import json
import os
from pathlib import Path
from typing import Any

import httpx


def _load_dotenv() -> None:
    """Load environment variables from a .env file if present."""
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
NEMOTRON_MODEL = os.getenv("NEMOTRON_MODEL", "nemotron-3-nano:30b-cloud")


class NemotronClient:
    """Lightweight client for Nemotron 3 Nano via Ollama's OpenAI-compatible API."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or NEMOTRON_MODEL
        self.base_url = (base_url or OLLAMA_HOST).rstrip("/")
        self.api_url = f"{self.base_url}/v1/chat/completions"
        self.client = httpx.Client(timeout=120.0)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Send a chat completion request with optional tool calling.

        Nemotron 3 Nano supports native tool calling in qwen3_coder format
        through Ollama's OpenAI-compatible endpoint.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        response = self.client.post(self.api_url, json=payload)
        response.raise_for_status()
        return response.json()

    def extract_response(self, completion: dict[str, Any]) -> dict[str, Any]:
        """Extract the assistant message from a completion response."""
        choice = completion["choices"][0]
        message = choice["message"]
        return {
            "content": message.get("content", ""),
            "tool_calls": message.get("tool_calls", []),
            "finish_reason": choice.get("finish_reason", "stop"),
        }

    def health_check(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = self.client.get(f"{self.base_url}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check if our model (or its base) is available
            model_base = self.model.split(":")[0]
            return any(model_base in m for m in models)
        except Exception:
            return False
