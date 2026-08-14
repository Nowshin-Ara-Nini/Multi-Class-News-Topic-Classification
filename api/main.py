"""
News Topic Classification — FastAPI REST API

Provides REST endpoints for news headline classification.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Docs:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

try:
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address
except ImportError:  # Allows imports in minimal/offline environments.
    Limiter = None


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Require ``X-API-Key`` only when API_KEY is configured."""
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Request model for single text prediction."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="News headline text to classify",
        examples=["Apple reports record quarterly revenue beating Wall Street expectations"],
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch text prediction."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of news headline texts to classify (max 100)",
    )


class PredictionResult(BaseModel):
    """Response model for a single prediction."""
    text: str = Field(description="Input text")
    predicted_class: str = Field(description="Predicted topic class")
    class_id: int = Field(description="Predicted class index")
    confidence: float = Field(description="Prediction confidence score")
    probabilities: Dict[str, float] = Field(description="Per-class probabilities")
    inference_time_ms: float = Field(description="Inference time in milliseconds")


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    success: bool = True
    result: PredictionResult


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    success: bool = True
    results: List[PredictionResult]
    total_time_ms: float = Field(description="Total processing time")
    count: int = Field(description="Number of predictions")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = "healthy"
    model_loaded: bool
    model_name: Optional[str] = None
    device: Optional[str] = None
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Response model for errors."""
    success: bool = False
    error: str
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

class AppState:
    """Application state container."""
    classifier: Any = None
    model_name: str = ""
    device: str = ""


state = AppState()


def _default_model_dir() -> Optional[Path]:
    """Return the newest complete model artifact directory, if one exists."""
    root = PROJECT_ROOT / "saved_models"
    if not root.exists():
        return None
    candidates = [
        directory for directory in root.iterdir()
        if directory.is_dir()
        and ((directory / "model.pt").exists() or (directory / "model.joblib").exists())
        and (directory / "pipeline_config.json").exists()
        and (directory / "preprocessor.joblib").exists()
        and (directory / "label_encoder.joblib").exists()
    ]
    return max(candidates, key=lambda directory: directory.stat().st_mtime) if candidates else None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    # Startup
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir is None:
        default_model = _default_model_dir()
        model_dir = str(default_model) if default_model is not None else ""

    if model_dir and Path(model_dir).exists():
        try:
            from src.inference import NewsClassifier
            state.classifier = NewsClassifier.from_checkpoint(model_dir)
            state.model_name = Path(model_dir).name
            state.device = str(state.classifier.device)
            logger.info(f"Model loaded from {model_dir}")
        except Exception as e:
            logger.warning(f"Could not load model from {model_dir}: {e}")
            logger.info("API running without model. Set MODEL_DIR env var to model path.")
    else:
        logger.warning(
            "No complete model artifact found. Train a model first or set MODEL_DIR."
        )

    yield

    # Shutdown
    state.classifier = None
    logger.info("API shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="News Topic Classification API",
    description=(
        "REST API for classifying news headlines into topic categories: "
        "Business, Science & Technology, Sports, and World News. "
        "Uses deep learning models trained on 88,000+ headlines."
    ),
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)

if Limiter is not None:
    limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(_request: Request, _exc: RateLimitExceeded):
        return JSONResponse(status_code=429, content={"success": False, "error": "rate_limit_exceeded"})


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or os.urandom(8).hex()
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info("request_id=%s method=%s path=%s status=%s duration_ms=%.2f", request_id, request.method, request.url.path, response.status_code, (time.perf_counter() - start) * 1000)
    return response

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check API health and model status.

    Returns the current status of the API and whether a model is loaded.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=state.classifier is not None,
        model_name=state.model_name or None,
        device=state.device or None,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    responses={
        503: {
            "model": ErrorResponse,
            "description": "Model not loaded",
        },
    },
)
async def predict(request: PredictionRequest, _auth: None = Depends(require_api_key)):
    """
    Classify a single news headline.

    Returns the predicted topic class with confidence score and
    per-class probability distribution.
    """
    if state.classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model loaded. Set MODEL_DIR environment variable to model path.",
        )

    try:
        start = time.perf_counter()
        result = state.classifier.predict(request.text)
        elapsed = (time.perf_counter() - start) * 1000

        return PredictionResponse(
            result=PredictionResult(
                text=request.text,
                predicted_class=result["class"],
                class_id=result["class_id"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                inference_time_ms=round(elapsed, 2),
            )
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/batch_predict",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    responses={
        503: {
            "model": ErrorResponse,
            "description": "Model not loaded",
        },
    },
)
async def batch_predict(request: BatchPredictionRequest, _auth: None = Depends(require_api_key)):
    """
    Classify a batch of news headlines (max 100).

    Returns predictions for all input texts with individual and total timing.
    """
    if state.classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model loaded. Set MODEL_DIR environment variable to model path.",
        )

    try:
        start = time.perf_counter()
        results = state.classifier.batch_predict(request.texts)
        elapsed = (time.perf_counter() - start) * 1000

        prediction_results = [
            PredictionResult(
                text=r["text"],
                predicted_class=r["class"],
                class_id=r["class_id"],
                confidence=r["confidence"],
                probabilities=r["probabilities"],
                inference_time_ms=r.get("inference_time_ms", round(elapsed / len(results), 2)),
            )
            for r in results
        ]

        return BatchPredictionResponse(
            results=prediction_results,
            total_time_ms=round(elapsed, 2),
            count=len(results),
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get("/models", tags=["System"])
async def list_available_models():
    """List all available trained models."""
    saved_models_dir = PROJECT_ROOT / "saved_models"
    models = []

    if saved_models_dir.exists():
        for model_dir in saved_models_dir.iterdir():
            if model_dir.is_dir():
                has_pt = (model_dir / "model.pt").exists()
                has_joblib = (model_dir / "model.joblib").exists()
                is_complete = (
                    (model_dir / "pipeline_config.json").exists()
                    and (model_dir / "preprocessor.joblib").exists()
                    and (model_dir / "label_encoder.joblib").exists()
                )
                if (has_pt or has_joblib) and is_complete:
                    models.append({
                        "name": model_dir.name,
                        "type": "pytorch" if has_pt else "sklearn",
                        "path": str(model_dir),
                    })

    return {"models": models, "current": state.model_name}


@app.get("/model/info", tags=["System"])
async def model_info():
    """Return model metadata when the selected model has a model card."""
    model_dir = Path(os.environ["MODEL_DIR"]) if os.environ.get("MODEL_DIR") else _default_model_dir()
    if model_dir is None:
        return {"model_name": None, "loaded": False}
    card = model_dir / "model_card.json"
    if card.exists():
        import json
        return json.loads(card.read_text(encoding="utf-8"))
    return {"model_name": state.model_name or None, "loaded": state.classifier is not None}


@app.get("/model/version", tags=["System"])
async def model_version():
    """Return the deployed model name and API version."""
    return {"api_version": app.version, "model_name": state.model_name or None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
