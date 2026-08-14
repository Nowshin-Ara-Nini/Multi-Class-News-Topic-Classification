from __future__ import annotations

import pytest


def test_health_endpoint_returns_200(monkeypatch, tmp_path):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from api.main import app, state

    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "missing-model"))
    state.classifier = None
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is False
