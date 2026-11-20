from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_doctor
from app.database.session import get_db
from app.models.case import CaseRecord
from app.models.image import ImageRecord
from app.models.prediction import PredictionRecord
from app.models.user import User
from app.schemas.case import CaseCreate, CaseRead, ClinicalReviewCreate
from app.services.case_service import (
    create_case,
    get_case_by_id,
    list_cases_for_user,
    record_clinical_review,
)

router = APIRouter(prefix="/cases", tags=["cases"])


@router.post("", response_model=CaseRead, status_code=status.HTTP_201_CREATED)
def create_case_endpoint(
    payload: CaseCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CaseRecord:
    case = create_case(db, current_user.id, payload.title, payload.description)
    return case


@router.get("", response_model=list[CaseRead])
def list_cases(
    doctor_view: bool = Query(False, description="Whether doctor is accessing the triage list"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[CaseRecord]:
    return list_cases_for_user(db, current_user, doctor_view=doctor_view)


@router.get("/{case_id}", response_model=CaseRead)
def get_case(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CaseRecord:
    case = get_case_by_id(db, current_user, case_id)
    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found or access denied",
        )
    return case


@router.post("/{case_id}/review", response_model=CaseRead)
def submit_clinical_review(
    case_id: int,
    payload: ClinicalReviewCreate,
    doctor_user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
) -> CaseRecord:
    case = record_clinical_review(
        db,
        case_id=case_id,
        doctor_id=doctor_user.id,
        review_status=payload.review_status,
        notes=payload.doctor_notes,
    )
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    return case


@router.get("/{case_id}/history")
def get_case_history(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    case = get_case_by_id(db, current_user, case_id)
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found or access denied")

    predictions = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.case_id == case_id)
        .order_by(PredictionRecord.created_at.desc())
        .all()
    )
    images = (
        db.query(ImageRecord)
        .filter(ImageRecord.case_id == case_id)
        .order_by(ImageRecord.created_at.desc())
        .all()
    )

    return {
        "case_id": case.id,
        "title": case.title,
        "review_status": case.review_status,
        "doctor_review_notes": case.doctor_review_notes,
        "total_images": len(images),
        "total_predictions": len(predictions),
        "predictions": [
            {
                "id": p.id,
                "image_id": p.image_id,
                "model_version": p.model_version,
                "prediction_label": p.prediction_label,
                "confidence": p.confidence,
                "risk_level": p.risk_level,
                "status": p.status,
                "created_at": p.created_at,
            }
            for p in predictions
        ],
        "images": [
            {
                "id": img.id,
                "filename": img.filename,
                "created_at": img.created_at,
                "status": img.status,
                "analysis": img.analysis,
            }
            for img in images
        ],
    }
