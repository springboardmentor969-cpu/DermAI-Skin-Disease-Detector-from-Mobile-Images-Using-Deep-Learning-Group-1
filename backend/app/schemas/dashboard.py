"""
Pydantic schemas for Clinician Analytics Dashboard.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class DashboardStatsResponse(BaseModel):
    total_cases: int
    cases_this_week: int
    cases_this_month: int
    avg_confidence: float
    high_risk_count: int
    pending_review_count: int


class ClassDistributionItem(BaseModel):
    class_name: str
    class_code: Optional[str] = None
    count: int
    percentage: float


class ConfidenceDistributionItem(BaseModel):
    bucket: str
    count: int


class RecentCaseItem(BaseModel):
    case_id: int
    patient_name: str
    predicted_class: str
    predicted_class_name: Optional[str] = None
    confidence: float
    risk_level: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CasesOverTimeItem(BaseModel):
    date: str
    count: int

