"""
Patient prediction and screening history router.
Provides paginated diagnostic history for patient timelines and clinical tracking.
"""

from typing import List, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models.case import Case
from app.models.image import Image
from app.models.patient import Patient
from app.models.prediction import Prediction
from app.models.user import User, UserRole
from app.schemas.history import PredictionHistoryItem
from app.services.risk_scorer import compute_risk
from app.services.storage_service import storage_service

router = APIRouter(tags=["Prediction History"])

CLASS_NAMES = {
    "mel": "Melanoma",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis / Bowen's Disease",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}


@router.get(
    "/patients/{patient_id}/history",
    response_model=Union[List[PredictionHistoryItem], dict],
    summary="Get patient prediction history",
    description=(
        "Retrieves a timeline of past AI analyses and predictions for a patient. "
        "Patients may only access their own history; clinicians and admins may access any patient's history."
    ),
)
def get_patient_history(
    patient_id: str,
    response: Response,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=50, description="Items per page (maximum 50)"),
    envelope: bool = Query(False, description="Whether to return { items: [], total: 0 } envelope"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Enforce maximum 50 results per page for query performance
    limit = min(max(1, limit), 50)
    is_me = str(patient_id).strip().lower() == "me"

    pid: Optional[int] = None
    if not is_me:
        try:
            pid = int(patient_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid patient_id: '{patient_id}'. Must be an integer or 'me'.",
            )

        # Patient role permission check
        if current_user.role == UserRole.PATIENT:
            # Verify patient ownership: case created_by == current_user.id or patient_id == current_user.id
            has_case = (
                db.query(Case)
                .filter(Case.patient_id == pid, Case.created_by == current_user.id)
                .first()
            )
            if not has_case and current_user.id != pid:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access forbidden: Patients can only view their own prediction history.",
                )

        # Check if patient exists - if not found, return empty records with HTTP 200 instead of 404
        patient = db.query(Patient).filter(Patient.id == pid).first()
        if not patient:
            response.headers["X-Total-Count"] = "0"
            response.headers["X-Page"] = str(page)
            response.headers["X-Limit"] = str(limit)
            if envelope:
                return {"items": [], "total": 0}
            return []

    # Build query joining Prediction -> Image -> Case
    query = (
        db.query(Prediction, Image, Case)
        .join(Image, Prediction.image_id == Image.id)
        .join(Case, Image.case_id == Case.id)
    )

    if is_me:
        query = query.filter(Case.created_by == current_user.id)
    else:
        query = query.filter(Case.patient_id == pid)
        if current_user.role == UserRole.PATIENT:
            query = query.filter(Case.created_by == current_user.id)

    total_count = query.count()
    offset = (page - 1) * limit
    results = (
        query.order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    # Attach pagination headers
    response.headers["X-Total-Count"] = str(total_count)
    response.headers["X-Page"] = str(page)
    response.headers["X-Limit"] = str(limit)

    if total_count == 0 and envelope:
        return {"items": [], "total": 0}

    items: List[PredictionHistoryItem] = []
    for pred, img, cs in results:
        # Resolve thumbnail URL from MinIO presigned URL
        thumb_url = storage_service.get_presigned_url(img.storage_key)
        # Direct API content URL
        content_url = f"/cases/{cs.id}/images/{img.id}/content"

        # Calculate risk level if not persisted
        risk_lvl = pred.risk_level
        if not risk_lvl:
            risk_info = compute_risk(pred.predicted_class, pred.confidence)
            risk_lvl = risk_info["risk_level"]

        cls_code = pred.predicted_class.lower()
        cls_name = CLASS_NAMES.get(cls_code, pred.predicted_class)

        items.append(
            PredictionHistoryItem(
                id=pred.id,
                case_id=cs.id,
                image_id=img.id,
                thumbnail_url=thumb_url,
                image_url=content_url,
                predicted_class=pred.predicted_class,
                predicted_class_name=cls_name,
                confidence=round(float(pred.confidence), 4),
                risk_level=risk_lvl,
                predictions=pred.top_n or [],
                created_at=pred.created_at,
            )
        )

    return items

