"""
Pydantic schemas for patient prediction history.
"""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict


class PredictionHistoryItem(BaseModel):
    id: int
    case_id: int
    image_id: int
    thumbnail_url: str
    image_url: Optional[str] = None
    predicted_class: str
    predicted_class_name: Optional[str] = None
    confidence: float
    risk_level: Optional[str] = None
    predictions: Optional[Any] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

