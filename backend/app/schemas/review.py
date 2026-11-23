"""
Clinical review schemas for expert diagnostic reviews, pending triage queues, and clinician metrics.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ClinicalReviewCreate(BaseModel):
    notes: Optional[str] = Field(None, description="Clinical evaluation and dermoscopy notes")
    confirmed_class: str = Field(..., max_length=10, description="Confirmed disease class code (e.g. mel, nv, bcc)")
    overridden: bool = Field(False, description="Whether the clinician overrode the AI's top prediction")


class ClinicalReviewResponse(BaseModel):
    id: int
    image_id: int
    clinician_id: int
    notes: Optional[str] = None
    confirmed_class: str
    confirmed_class_name: Optional[str] = None
    overridden: bool = False
    created_at: datetime
    clinician_name: Optional[str] = None

    class Config:
        from_attributes = True


class PendingReviewItem(BaseModel):
    image_id: int
    case_id: int
    patient_id: Optional[int] = None
    patient_name: str
    captured_at: datetime
    storage_key: str
    predicted_class: Optional[str] = None
    predicted_class_name: Optional[str] = None
    confidence: Optional[float] = None
    risk_level: Optional[str] = None
    top_n: Optional[List[Dict[str, Any]]] = None


class ClinicianStatsResponse(BaseModel):
    total_cases_reviewed: int
    pending_review: int
    ai_accuracy: float
    overrides_this_week: int


class ClinicianCaseItem(BaseModel):
    image_id: int
    case_id: int
    patient_name: str
    storage_key: str
    predicted_class: str
    predicted_class_name: str
    confidence: float
    risk_level: str
    date: datetime
    reviewed: bool
    confirmed_class: Optional[str] = None
    confirmed_class_name: Optional[str] = None
    overridden: bool = False
    notes: Optional[str] = None
    review_id: Optional[int] = None
