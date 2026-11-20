from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CaseCreate(BaseModel):
    title: str = Field(..., min_length=2, max_length=255)
    description: str | None = None
    status: str = "new"


class ClinicalReviewCreate(BaseModel):
    review_status: str = Field(..., description="e.g. reviewed, biopsy_recommended, benign_observation, urgent_consult")
    doctor_notes: str = Field(..., min_length=2, max_length=2000)


class PredictionSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), from_attributes=True)

    id: int
    case_id: int
    image_id: int | None = None
    model_version: str
    prediction_label: str | None = None
    confidence: float | None = None
    risk_level: str | None = None
    status: str
    created_at: datetime


class CaseImageSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    case_id: int
    filename: str
    content_type: str
    file_size: int
    width: int | None = None
    height: int | None = None
    status: str
    created_at: datetime
    analysis: dict[str, Any] | None = None


class CaseRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str | None = None
    status: str
    doctor_review_notes: str | None = None
    reviewed_by_id: int | None = None
    review_status: str | None = "pending_review"
    created_at: datetime
    updated_at: datetime
    user_id: int
    images: list[CaseImageSummary] = Field(default_factory=list)
    predictions: list[PredictionSummary] = Field(default_factory=list)
