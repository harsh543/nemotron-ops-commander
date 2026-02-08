"""StackOverflow public data connector."""

from __future__ import annotations

from typing import List, Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from knowledge_connectors.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)

STACKOVERFLOW_API = "https://api.stackexchange.com/2.3/search/advanced"
STACKOVERFLOW_ANSWERS_API = "https://api.stackexchange.com/2.3/questions/{ids}/answers"


class StackOverflowSnippet(dict):
    """Lightweight container for StackOverflow snippets."""


class StackOverflowConnector:
    """Connector for StackOverflow public search."""

    def __init__(self) -> None:
        self.breaker = CircuitBreaker()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def search(self, query: str, tags: List[str] | None = None, top_k: int = 10) -> List[str]:
        """Search StackOverflow and return question URLs."""

        if not self.breaker.allow_request():
            raise RuntimeError("StackOverflow circuit breaker open")

        params = {
            "q": query,
            "order": "desc",
            "sort": "relevance",
            "site": "stackoverflow",
            "pagesize": top_k,
        }
        if tags:
            params["tagged"] = ";".join(tags)

        try:
            with httpx.Client(timeout=15) as client:
                response = client.get(STACKOVERFLOW_API, params=params)
                response.raise_for_status()
                data = response.json()
                self.breaker.record_success()
                return [item["link"] for item in data.get("items", [])]
        except Exception as exc:  # noqa: BLE001
            self.breaker.record_failure()
            raise exc

    def search_error_signatures(
        self, signature: str, top_k: int = 5
    ) -> List[StackOverflowSnippet]:
        """Search error signatures and return accepted answers when available."""

        params = {
            "q": signature,
            "order": "desc",
            "sort": "relevance",
            "site": "stackoverflow",
            "pagesize": top_k,
        }
        if not self.breaker.allow_request():
            raise RuntimeError("StackOverflow circuit breaker open")

        try:
            with httpx.Client(timeout=15) as client:
                response = client.get(STACKOVERFLOW_API, params=params)
                response.raise_for_status()
                data = response.json()
                question_ids = [str(item["question_id"]) for item in data.get("items", [])]
                snippets: List[StackOverflowSnippet] = []
                if not question_ids:
                    self.breaker.record_success()
                    return snippets

                answers = self._fetch_accepted_answers(client, question_ids)
                for item in data.get("items", []):
                    qid = str(item["question_id"])
                    accepted = answers.get(qid)
                    snippets.append(
                        StackOverflowSnippet(
                            source="stackoverflow",
                            title=item.get("title"),
                            url=item.get("link"),
                            accepted_answer=accepted,
                            tags=item.get("tags", []),
                        )
                    )
                self.breaker.record_success()
                return snippets
        except Exception as exc:  # noqa: BLE001
            self.breaker.record_failure()
            raise exc

    def _fetch_accepted_answers(
        self, client: httpx.Client, question_ids: List[str]
    ) -> dict:
        """Fetch accepted answers for a list of question IDs."""

        ids = ";".join(question_ids)
        params = {
            "order": "desc",
            "sort": "activity",
            "site": "stackoverflow",
            "filter": "withbody",
        }
        response = client.get(STACKOVERFLOW_ANSWERS_API.format(ids=ids), params=params)
        response.raise_for_status()
        data = response.json()
        accepted: dict[str, Optional[str]] = {qid: None for qid in question_ids}
        for item in data.get("items", []):
            if item.get("is_accepted"):
                qid = str(item.get("question_id"))
                accepted[qid] = item.get("body")
        return accepted
