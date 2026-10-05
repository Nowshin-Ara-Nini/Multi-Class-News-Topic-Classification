"""Pinned, CPU-only inference API for local use and Vercel."""
from __future__ import annotations
import hmac
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StringConstraints

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)]

class PredictionRequest(BaseModel):
    text: Text

class BatchPredictionRequest(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=100)

class PredictionResult(BaseModel):
    text: str
    predicted_class: str
    class_id: int
    confidence: float
    probabilities: dict[str, float]
    inference_time_ms: float

class PredictionResponse(BaseModel):
    success: bool = True
    result: PredictionResult

class BatchPredictionResponse(BaseModel):
    success: bool = True
    results: list[PredictionResult]
    count: int
    total_time_ms: float

class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    model_name: Optional[str] = None
    device: Optional[str] = None
    version: str = "2.0.0"

class AppState:
    classifier = None
    model_name = ""
    device = "cpu"
    model_dir = None
    card = {}
    error = None

state = AppState()
load_lock = threading.Lock()
inference_lock = threading.Lock()


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.getenv("API_KEY")
    if os.getenv("VERCEL") and not expected:
        raise HTTPException(503, "Backend API key is not configured")
    if expected and not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(401, "Invalid or missing API key")


def load_model():
    with load_lock:
        if state.classifier is not None:
            return
        model_dir = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "model"))).resolve()
        state.model_dir = model_dir
        try:
            if not (model_dir / "pipeline_config.json").is_file():
                raise FileNotFoundError("Set MODEL_DIR to one complete exported model")
            import torch
            torch.set_num_threads(1)
            from src.inference import NewsClassifier
            from src.release import artifact_digest
            card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
            if os.getenv("VERCEL"):
                fingerprint = artifact_digest(model_dir)
                benchmark = card.get("deployment_benchmark") or {}
                if not benchmark.get("passed") or benchmark.get("artifact_sha256") != fingerprint:
                    raise ValueError("Artifact has no matching successful deployment benchmark")
                if (card.get("test_metrics") or {}).get("artifact_sha256") != fingerprint:
                    raise ValueError("Artifact has no matching final test evaluation")
            state.classifier = NewsClassifier.from_checkpoint(model_dir, device="cpu")
            state.classifier.config.inference.batch_size = 10
            state.card = card
            state.model_name = card.get("model_name", model_dir.name)
            state.error = None
        except Exception:
            state.error = "Model could not be loaded"
            logger.exception("Pinned model load failed")


@asynccontextmanager
async def lifespan(app):
    # Some function adapters do not invoke lifespan; handlers also lazily load.
    yield
    state.classifier = None


app = FastAPI(title="News Topic Classification", version="2.0.0", lifespan=lifespan)
origins = [s.strip() for s in os.getenv("ALLOWED_ORIGINS", "").split(",") if s.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-API-Key"])


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(model_loaded=state.classifier is not None, model_name=state.model_name or None,
                          device=state.device)


def ready_classifier():
    if state.classifier is None:
        load_model()
    if state.classifier is None:
        raise HTTPException(503, state.error or "Model is starting")
    return state.classifier


@app.get("/ready")
def readiness(_auth=Depends(require_api_key)):
    ready_classifier()
    return {"ready": True, "model_name": state.model_name}


def classify(texts):
    classifier = ready_classifier()
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(429, "Model is busy; please retry shortly", headers={"Retry-After": "2"})
    try:
        start = time.perf_counter()
        results = classifier.batch_predict(texts)
        elapsed = (time.perf_counter() - start) * 1000
        predictions = [PredictionResult(text=r["text"], predicted_class=r["class"], class_id=r["class_id"],
                        confidence=r["confidence"], probabilities=r["probabilities"],
                        inference_time_ms=r.get("inference_time_ms", elapsed / len(results))) for r in results]
        return predictions, elapsed
    finally:
        # Avoid retaining user headlines indefinitely in the preprocessing cache.
        if hasattr(classifier.preprocessor, "_cache"):
            classifier.preprocessor._cache.clear()
        inference_lock.release()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, _auth=Depends(require_api_key)):
    rows, _ = classify([request.text])
    return PredictionResponse(result=rows[0])


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest, _auth=Depends(require_api_key)):
    rows, elapsed = classify(request.texts)
    return BatchPredictionResponse(results=rows, count=len(rows), total_time_ms=elapsed)


@app.get("/model/info")
def model_info(_auth=Depends(require_api_key)):
    ready_classifier()
    # Expose display metadata, not filesystem paths or the training environment.
    keys = ("model_name", "architecture", "feature_type", "preprocessing_mode", "classes",
            "parameter_count", "artifact_sha256", "quantized", "test_metrics")
    result = {key: state.card.get(key) for key in keys}
    summary = state.card.get("training_summary", {})
    result["training_summary"] = {k: summary.get(k) for k in ("best_val_f1", "best_val_acc", "best_epoch", "seed")}
    return result


@app.get("/model/version")
def model_version(_auth=Depends(require_api_key)):
    ready_classifier()
    return {"api_version": app.version, "model_name": state.model_name,
            "artifact_sha256": state.card.get("artifact_sha256")}


@app.get("/models")
def models(_auth=Depends(require_api_key)):
    ready_classifier()
    path = state.model_dir / "comparison.json"
    return {"current": state.model_name, "models": json.loads(path.read_text()) if path.exists() else []}
