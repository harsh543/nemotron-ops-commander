"""GitHub public issues search connector (unauthenticated)."""

from __future__ import annotations

from typing import List

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from knowledge_connectors.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)

GITHUB_SEARCH_API = "https://api.github.com/search/issues"


class GitHubPublicIssuesConnector:
    """Connector for GitHub public issues search without auth."""

    def __init__(self) -> None:
        self.breaker = CircuitBreaker()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def search(self, query: str, top_k: int = 10) -> List[str]:
        """Search GitHub issues and return issue URLs."""

        if not self.breaker.allow_request():
            raise RuntimeError("GitHub circuit breaker open")

        params = {"q": query, "per_page": top_k}
        try:
            with httpx.Client(timeout=15) as client:
                response = client.get(GITHUB_SEARCH_API, params=params)
                response.raise_for_status()
                data = response.json()
                self.breaker.record_success()
                return [item["html_url"] for item in data.get("items", [])]
        except Exception as exc:  # noqa: BLE001
            self.breaker.record_failure()
            raise exc

    def search_error_signatures(self, signature: str, top_k: int = 5) -> List[str]:
        """Search GitHub public issues for error signatures."""

        return self.search(signature, top_k=top_k)
