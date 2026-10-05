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


def test_authenticated_api_rejects_invalid_key_before_loading(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    monkeypatch.setenv("API_KEY", "expected-key")
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "A news headline"})
    assert response.status_code == 401


def test_batch_validates_each_headline(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    monkeypatch.setenv("API_KEY", "expected-key")
    with TestClient(app) as client:
        response = client.post("/batch_predict", headers={"X-API-Key": "expected-key"},
                               json={"texts": ["valid headline", "   "]})
    assert response.status_code == 422
