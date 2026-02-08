"""Public knowledge ingestion pipeline (read-only)."""

from __future__ import annotations

from typing import Dict, List

import structlog

from knowledge_connectors.cloud_docs_connector import CloudDocsConnector
from knowledge_connectors.github_public_issues import GitHubPublicIssuesConnector
from knowledge_connectors.k8s_docs_connector import KubernetesDocsConnector
from knowledge_connectors.stackoverflow_connector import StackOverflowConnector

logger = structlog.get_logger(__name__)


class KnowledgeIngestion:
    """Aggregate public knowledge sources for RAG enrichment."""

    def __init__(self) -> None:
        self.stackoverflow = StackOverflowConnector()
        self.k8s_docs = KubernetesDocsConnector()
        self.cloud_docs = CloudDocsConnector()
        self.github = GitHubPublicIssuesConnector()

    def collect(self, error_signatures: List[str]) -> Dict[str, List]:
        """Collect read-only knowledge snippets and URLs.

        Returns a dict with keys:
        - stackoverflow
        - kubernetes
        - aws
        - azure
        - github
        """

        stack_snippets = []
        for signature in error_signatures:
            stack_snippets.extend(self.stackoverflow.search_error_signatures(signature))

        k8s_urls = self.k8s_docs.index_failure_modes()
        k8s_urls.extend(self.k8s_docs.index_upgrade_regressions())

        aws_urls = self.cloud_docs.ingest_aws_troubleshooting()
        azure_urls = self.cloud_docs.ingest_azure_troubleshooting()

        github_urls = []
        for signature in error_signatures:
            github_urls.extend(self.github.search_error_signatures(signature))

        logger.info(
            "knowledge.collect",
            stackoverflow=len(stack_snippets),
            kubernetes=len(k8s_urls),
            aws=len(aws_urls),
            azure=len(azure_urls),
            github=len(github_urls),
        )

        return {
            "stackoverflow": stack_snippets,
            "kubernetes": k8s_urls,
            "aws": aws_urls,
            "azure": azure_urls,
            "github": github_urls,
        }
