"""Pydantic request/response schemas for the FakeGuard API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, description="News article text to classify")
    model: str | None = Field(
        default="ensemble",
        description="Model to use: tfidf_lr | lstm | bilstm | bert | ensemble",
    )

    model_config = {"json_schema_extra": {"example": {"text": "Scientists discover new vaccine...", "model": "ensemble"}}}


class WordImportance(BaseModel):
    word: str
    score: float = Field(description="Importance score in [-1, 1]. Positive → FAKE evidence, negative → REAL evidence")
    fake_contribution: bool = Field(description="True if this word pushes prediction toward FAKE")


class PredictResponse(BaseModel):
    label: str = Field(description="'FAKE' or 'REAL'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for the predicted class")
    probabilities: dict[str, float] = Field(description="Raw class probabilities")
    word_importance: list[WordImportance] = Field(description="Top influential words")
    model_used: str = Field(description="Name of the model that produced this result")
    char_count: int = Field(description="Number of characters in input text")
    word_count: int = Field(description="Number of words in input text")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)
    model: str | None = "ensemble"


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    count: int


class ModelMetric(BaseModel):
    name: str
    display_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    status: str


class ModelsListResponse(BaseModel):
    models: list[ModelMetric]


class ModelInfoResponse(BaseModel):
    models: list[dict[str, Any]]
