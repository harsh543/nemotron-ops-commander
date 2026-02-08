"""Kubernetes documentation & release notes connector."""

from __future__ import annotations

from typing import List

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from knowledge_connectors.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)

K8S_SITE_SEARCH = "https://kubernetes.io/search/"
K8S_RELEASE_NOTES = "https://kubernetes.io/docs/setup/release/notes/"


class KubernetesDocsConnector:
    """Connector for Kubernetes documentation search and release notes."""

    def __init__(self) -> None:
        self.breaker = CircuitBreaker()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def search_docs(self, query: str, top_k: int = 10) -> List[str]:
        """Search Kubernetes docs and return result URLs."""

        if not self.breaker.allow_request():
            raise RuntimeError("Kubernetes docs circuit breaker open")

        params = {"q": query}
        try:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                response = client.get(K8S_SITE_SEARCH, params=params)
                response.raise_for_status()
                text = response.text
                self.breaker.record_success()
                urls = []
                for line in text.split("\n"):
                    if "href=\"/docs/" in line:
                        start = line.find("href=\"") + 6
                        end = line.find("\"", start)
                        href = line[start:end]
                        if href.startswith("/docs/"):
                            urls.append(f"https://kubernetes.io{href}")
                return urls[:top_k]
        except Exception as exc:  # noqa: BLE001
            self.breaker.record_failure()
            raise exc

    def release_notes_url(self) -> str:
        """Return the base release notes URL."""

        return K8S_RELEASE_NOTES

    def index_failure_modes(self, top_k: int = 10) -> List[str]:
        """Return common failure mode docs for ingestion."""

        queries = [
            "CrashLoopBackOff",
            "ImagePullBackOff",
            "Readiness probe failed",
            "Node NotReady",
            "Evicted pods",
            "OOMKilled",
            "Ingress 502",
        ]
        urls: List[str] = []
        for query in queries:
            urls.extend(self.search_docs(query, top_k=2))
            if len(urls) >= top_k:
                break
        return urls[:top_k]

    def index_upgrade_regressions(self) -> List[str]:
        """Return release notes URL for upgrade issues and regressions."""

        return [K8S_RELEASE_NOTES]
