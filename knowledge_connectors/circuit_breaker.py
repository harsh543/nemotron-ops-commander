"""Simple circuit breaker implementation."""

from __future__ import annotations

import time


class CircuitBreaker:
    """Naive in-memory circuit breaker."""

    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 30) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.opened_at: float | None = None

    def allow_request(self) -> bool:
        if self.opened_at is None:
            return True
        if time.time() - self.opened_at > self.reset_timeout:
            self.failures = 0
            self.opened_at = None
            return True
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.time()
