"""Prometheus metrics helpers."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "nemotron_requests_total",
    "Total number of requests processed",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "nemotron_request_latency_ms",
    "Request latency in milliseconds",
    ["endpoint"],
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000),
)


def record_request(endpoint: str, status: str, latency_ms: float) -> None:
    """Record metrics for a request."""

    REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_ms)
