"""
FakeGuard — Fake News Detection API
FastAPI backend serving predictions from ML models.
Endpoints:
  GET  /health          — service health check
  POST /predict         — single text prediction with word importance
  POST /predict/batch   — batch prediction
  GET  /models          — list available models and their metrics
  GET  /model-info      — detailed model information
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import fastapi
import fastapi.middleware.cors

# ── add backend root to Python path ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from app.predictor import FakeNewsPredictor, get_predictor
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ModelInfoResponse,
    ModelsListResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)


# ── lifespan: warm up models on startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    logger.info("Starting FakeGuard API — loading models…")
    predictor = get_predictor()
    predictor.load_or_train()
    logger.info("Models ready.")
    yield
    logger.info("Shutting down FakeGuard API.")


# ── app ───────────────────────────────────────────────────────────────────────
app = fastapi.FastAPI(
    title="FakeGuard API",
    description="AI-powered fake news detection using LSTM, BiLSTM and TF-IDF ensemble models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health() -> dict[str, Any]:
    """Service health check."""
    predictor = get_predictor()
    return {
        "status": "ok",
        "models_loaded": predictor.is_ready(),
        "available_models": predictor.available_models(),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict whether the given news text is FAKE or REAL.

    Returns:
    - label: 'FAKE' | 'REAL'
    - confidence: probability of predicted class (0-1)
    - probabilities: dict with scores for each class
    - word_importance: list of (word, score) pairs highlighting key evidence words
    - model_used: name of the model that produced the result
    """
    predictor = get_predictor()
    result = predictor.predict(
        text=request.text,
        model_name=request.model or "ensemble",
    )
    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Predict fake/real for a batch of texts."""
    predictor = get_predictor()
    results = [
        PredictResponse(**predictor.predict(text=t, model_name=request.model or "ensemble"))
        for t in request.texts
    ]
    return BatchPredictResponse(results=results, count=len(results))


@app.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def list_models() -> ModelsListResponse:
    """List available models with their performance metrics."""
    predictor = get_predictor()
    return ModelsListResponse(models=predictor.model_metrics())


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Models"])
async def model_info() -> ModelInfoResponse:
    """Detailed information about the deployed models."""
    return ModelInfoResponse(
        models=[
            {
                "name": "tfidf_lr",
                "display_name": "TF-IDF + Logistic Regression",
                "description": (
                    "Classic NLP pipeline using TF-IDF vectorisation with character and word "
                    "n-grams fed into a regularised Logistic Regression classifier. Fast, "
                    "interpretable, and surprisingly competitive baseline."
                ),
                "type": "classical_ml",
                "features": ["tf-idf", "n-grams", "char-grams"],
            },
            {
                "name": "lstm",
                "display_name": "LSTM Neural Network",
                "description": (
                    "Long Short-Term Memory network with GloVe word embeddings. "
                    "Captures sequential dependencies and long-range context within news articles."
                ),
                "type": "deep_learning",
                "features": ["word-embeddings", "lstm", "attention"],
            },
            {
                "name": "bilstm",
                "display_name": "Bidirectional LSTM",
                "description": (
                    "Bidirectional LSTM that reads text both forward and backward, "
                    "capturing richer contextual representations. Generally outperforms "
                    "unidirectional LSTM on classification tasks."
                ),
                "type": "deep_learning",
                "features": ["word-embeddings", "bilstm", "attention"],
            },
            {
                "name": "bert",
                "display_name": "BERT Transformer",
                "description": (
                    "bert-base-uncased fine-tuned on the Kaggle Fake News dataset. "
                    "State-of-the-art transformer architecture with deep bidirectional "
                    "pre-training. Highest accuracy but requires GPU for best performance."
                ),
                "type": "transformer",
                "features": ["bert", "attention", "contextual-embeddings"],
            },
            {
                "name": "ensemble",
                "display_name": "Ensemble (Recommended)",
                "description": (
                    "Soft-voting ensemble combining TF-IDF LR, LSTM, and BiLSTM predictions. "
                    "Reduces model-specific errors and provides the best overall performance."
                ),
                "type": "ensemble",
                "features": ["voting", "calibrated-probabilities"],
            },
        ]
    )
