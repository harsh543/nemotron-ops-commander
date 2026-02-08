"""API tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_health():
    client = TestClient(app)
    response = client.get("/health/")
    assert response.status_code in (200, 401)
