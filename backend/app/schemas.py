from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr

VALID_ROLES = [
    "Patient / General User",
    "Dermatologist / Clinician",
    "Medical Researcher",
    "Data Scientist / ML Engineer",
    "Administrator",
]

class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str
    role: str = "Patient / General User"
    institution: Optional[str] = "DermAI Clinical Center"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    institution: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class PatientCreate(BaseModel):
    patient_code: Optional[str] = None
    age: int
    gender: str
    fitzpatrick_skin_type: Optional[str] = "Type III"
    medical_history: Optional[str] = "No prior melanoma family history"
    known_allergies: Optional[str] = "NKDA"

class CaseCreate(BaseModel):
    patient_code: str
    age: int
    gender: str
    medical_history: Optional[str] = "None"
    known_allergies: Optional[str] = "NKDA"
    body_location: str
    image_source: str = "Mobile Camera"
    clinical_notes: Optional[str] = ""

class QualityMetrics(BaseModel):
    width_px: int
    height_px: int
    file_size_kb: float
    format: str
    blur_laplacian_var: float
    brightness_mean: float
    contrast_std: float
    lesion_visibility_score: float
    quality_status: str
    suitable_for_ai: bool
    warnings: List[str]

class PreprocessingResult(BaseModel):
    image_code: str
    case_code: str
    quality_metrics: QualityMetrics
    pipeline_steps_applied: List[str]
    tensor_shape: List[int]
    normalization_stats: Dict[str, Any]
    raw_image_base64: str
    hair_removed_base64: str
    clahe_enhanced_base64: str
    normalized_output_base64: str
