"""AWS/Azure public troubleshooting docs connector."""

from __future__ import annotations

from typing import List

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from knowledge_connectors.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)

AWS_DOCS_SEARCH = "https://docs.aws.amazon.com/search/doc-search.html"
AZURE_DOCS_SEARCH = "https://learn.microsoft.com/en-us/search/"


class CloudDocsConnector:
    """Connector for AWS and Azure public troubleshooting docs."""

    def __init__(self) -> None:
        self.breaker = CircuitBreaker()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def search_aws(self, query: str, top_k: int = 10) -> List[str]:
        """Search AWS docs and return result URLs."""

        if not self.breaker.allow_request():
            raise RuntimeError("Cloud docs circuit breaker open")

        params = {"searchPath": "", "searchQuery": query, "this_doc_product": ""}
        try:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                response = client.get(AWS_DOCS_SEARCH, params=params)
                response.raise_for_status()
                text = response.text
                self.breaker.record_success()
                urls = []
                for line in text.split("\n"):
                    if "data-analytics-link=\"search-result-link\"" in line and "href=\"" in line:
                        start = line.find("href=\"") + 6
                        end = line.find("\"", start)
                        href = line[start:end]
                        if href.startswith("http"):
                            urls.append(href)
                return urls[:top_k]
        except Exception as exc:  # noqa: BLE001
            self.breaker.record_failure()
            raise exc

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def search_azure(self, query: str, top_k: int = 10) -> List[str]:
        """Search Azure docs and return result URLs."""

        if not self.breaker.allow_request():
            raise RuntimeError("Cloud docs circuit breaker open")

        params = {"terms": query}
        try:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                response = client.get(AZURE_DOCS_SEARCH, params=params)
                response.raise_for_status()
                text = response.text
                self.breaker.record_success()
                urls = []
                for line in text.split("\n"):
                    if "data-bi-name=\"search-result\"" in line and "href=\"" in line:
                        start = line.find("href=\"") + 6
                        end = line.find("\"", start)
                        href = line[start:end]
                        if href.startswith("http"):
                            urls.append(href)
                return urls[:top_k]
        except Exception as exc:  # noqa: BLE001
            self.breaker.record_failure()
            raise exc

    def ingest_aws_troubleshooting(self, top_k: int = 10) -> List[str]:
        """Return AWS troubleshooting URLs for ingestion."""

        topics = [
            "EKS troubleshooting",
            "EC2 troubleshooting",
            "IAM troubleshooting",
            "VPC networking troubleshooting",
        ]
        urls: List[str] = []
        for topic in topics:
            urls.extend(self.search_aws(topic, top_k=3))
            if len(urls) >= top_k:
                break
        return urls[:top_k]

    def ingest_azure_troubleshooting(self, top_k: int = 10) -> List[str]:
        """Return Azure troubleshooting URLs for ingestion."""

        topics = [
            "AKS troubleshooting",
            "Azure VM troubleshooting",
            "Azure identity troubleshooting",
            "Azure networking troubleshooting",
        ]
        urls: List[str] = []
        for topic in topics:
            urls.extend(self.search_azure(topic, top_k=3))
            if len(urls) >= top_k:
                break
        return urls[:top_k]
