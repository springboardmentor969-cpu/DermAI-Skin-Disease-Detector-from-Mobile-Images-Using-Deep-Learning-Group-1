"""
Pydantic schemas for Patients, Cases, Images, Predictions, and Audits.
"""

from datetime import datetime
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, ConfigDict


# ------------------------------------------------------------------------------
# Patient Schemas (Strictly de-identified demographic info)
# ------------------------------------------------------------------------------
class PatientBase(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    notes: Optional[str] = None


class PatientCreate(PatientBase):
    pass


class PatientResponse(PatientBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Prediction Schemas
# ------------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    id: int
    image_id: int
    model_version: str
    predicted_class: str
    confidence: float
    top_n: Optional[List[Dict[str, Any]]] = None
    created_at: datetime
    is_low_confidence: bool = False
    warning: Optional[str] = None
    demo_model: bool = False
    segmentation_overlay: Optional[str] = None
    disclaimer: str = "AI-assisted analysis. Not a medical diagnosis."
    ensemble: bool = False
    gradcam_key: Optional[str] = None
    risk_level: Optional[str] = None
    risk_color: Optional[str] = None
    risk_message: Optional[str] = None
    urgency_days: Optional[int] = None
    probabilities: Optional[Dict[str, float]] = None
    all_predictions: Optional[List[Dict[str, Any]]] = None
    num_models_used: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Image Schemas
# ------------------------------------------------------------------------------
class ImageResponse(BaseModel):
    id: int
    case_id: int
    storage_key: str
    source: str
    captured_at: datetime
    quality_label: Optional[str] = None
    quality_details: Optional[Dict[str, Any]] = None
    warning: Optional[str] = None
    predictions: List[PredictionResponse] = []

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Case Schemas
# ------------------------------------------------------------------------------
class CaseBase(BaseModel):
    body_location: str
    status: str = "submitted"


class CaseCreate(CaseBase):
    # Demographics for creating or associating patient record
    patient_id: Optional[int] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    clinical_notes: Optional[str] = None


class CaseResponse(CaseBase):
    id: int
    patient_id: int
    created_by: int
    created_at: datetime
    patient: Optional[PatientResponse] = None
    images: List[ImageResponse] = []

    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------------------
# Audit Log Schemas
# ------------------------------------------------------------------------------
class AuditLogResponse(BaseModel):
    id: int
    user_id: Optional[int] = None
    action: str
    details: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
