"""
Tests for Streamlit application helper functions and deployment logic.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

pytest.importorskip("streamlit", reason="Archived UI; Streamlit is not a production dependency")

from legacy.streamlit_app import (
    is_valid_checkpoint,
    check_deployment_health,
    get_best_model,
    build_leaderboard,
    extract_model_metadata,
    detect_deployment_mode,
)


@pytest.fixture
def dummy_checkpoint(tmp_path):
    """Create a mock valid TF-IDF checkpoint directory."""
    model_dir = tmp_path / "mock_lr_optimum_tfidf"
    model_dir.mkdir(parents=True)

    pipeline = {
        "feature_type": "tfidf",
        "model_name": "logistic_regression",
        "model_kwargs": {},
        "preprocessing_mode": "optimum",
        "class_names": ["Business", "Science and Technology", "Sports", "World News"],
    }
    (model_dir / "pipeline_config.json").write_text(json.dumps(pipeline), encoding="utf-8")
    (model_dir / "preprocessor.joblib").write_bytes(b"dummy")
    (model_dir / "label_encoder.joblib").write_bytes(b"dummy")
    (model_dir / "model.joblib").write_bytes(b"dummy")
    (model_dir / "vectorizer.joblib").write_bytes(b"dummy")

    model_card = {
        "model_name": "mock_lr_optimum_tfidf",
        "architecture": "logistic_regression",
        "feature_type": "tfidf",
        "preprocessing_mode": "optimum",
        "classes": ["Business", "Science and Technology", "Sports", "World News"],
        "parameter_count": None,
        "model_size_mb": 29.5,
        "training_summary": {
            "best_val_f1": 0.911,
            "best_val_acc": 0.9228,
        },
        "environment": {
            "captured_at_utc": "2026-08-31T13:48:49.489523+00:00",
        },
    }
    (model_dir / "model_card.json").write_text(json.dumps(model_card), encoding="utf-8")

    return model_dir


def test_is_valid_checkpoint_success(dummy_checkpoint):
    is_valid, reason = is_valid_checkpoint(dummy_checkpoint)
    assert is_valid is True
    assert reason is None


def test_is_valid_checkpoint_missing_file(dummy_checkpoint):
    (dummy_checkpoint / "vectorizer.joblib").unlink()
    is_valid, reason = is_valid_checkpoint(dummy_checkpoint)
    assert is_valid is False
    assert "vectorizer.joblib" in reason


def test_check_deployment_health(dummy_checkpoint):
    health = check_deployment_health(dummy_checkpoint)
    assert health["healthy"] is True
    assert health["message"] == "Deployment Healthy"
    assert health["details"]["Model Weights"] is True
    assert health["details"]["Pipeline Config"] is True
    assert health["details"]["Feature Extractor"] is True


def test_extract_model_metadata(dummy_checkpoint):
    meta = extract_model_metadata(dummy_checkpoint)
    assert meta["model_name"] == "mock_lr_optimum_tfidf"
    assert meta["architecture"] == "Logistic Regression"
    assert meta["training_date"] == "2026-08-31"
    assert "Linear model" in meta["parameter_count"]
    assert "92.3%" in meta["val_accuracy"]
    assert "91.1%" in meta["val_f1"]
    assert meta["num_classes"] == 4


def test_get_best_model(tmp_path, dummy_checkpoint):
    # Create second model with higher F1
    model_dir_2 = tmp_path / "mock_transformer"
    model_dir_2.mkdir(parents=True)
    pipeline = {
        "feature_type": "word2vec",
        "model_name": "transformer",
        "class_names": ["Business", "Science and Technology", "Sports", "World News"],
    }
    (model_dir_2 / "pipeline_config.json").write_text(json.dumps(pipeline), encoding="utf-8")
    (model_dir_2 / "preprocessor.joblib").write_bytes(b"dummy")
    (model_dir_2 / "label_encoder.joblib").write_bytes(b"dummy")
    (model_dir_2 / "model.pt").write_bytes(b"dummy")
    (model_dir_2 / "model_card.json").write_text(json.dumps({
        "model_name": "mock_transformer",
        "architecture": "transformer",
        "training_summary": {"best_val_f1": 0.935, "best_val_acc": 0.941},
        "environment": {"captured_at_utc": "2026-09-01T10:00:00+00:00"}
    }), encoding="utf-8")

    best = get_best_model([dummy_checkpoint, model_dir_2])
    assert best == model_dir_2

    leaderboard = build_leaderboard([dummy_checkpoint, model_dir_2], top_n=5)
    assert len(leaderboard) == 2
    assert leaderboard[0]["name"] == "mock_transformer"
    assert leaderboard[0]["val_f1"] == "93.5%"


def test_detect_deployment_mode_production(dummy_checkpoint, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(dummy_checkpoint))
    mode, pinned_path, candidates = detect_deployment_mode()
    assert mode == "production"
    assert pinned_path == dummy_checkpoint
    assert len(candidates) == 1


def test_metadata_graceful_fallback(tmp_path):
    """Test metadata extraction when model_card.json is missing."""
    sparse_dir = tmp_path / "sparse_model"
    sparse_dir.mkdir()
    pipeline = {
        "feature_type": "tfidf",
        "model_name": "dnn",
        "class_names": ["A", "B"],
    }
    (sparse_dir / "pipeline_config.json").write_text(json.dumps(pipeline), encoding="utf-8")
    meta = extract_model_metadata(sparse_dir)
    assert meta["architecture"] == "Deep Neural Network"
    assert meta["num_classes"] == 2
    assert meta["parameter_count"] == "N/A"
    assert meta["val_f1"] == "N/A"
    assert meta["val_accuracy"] == "N/A"
