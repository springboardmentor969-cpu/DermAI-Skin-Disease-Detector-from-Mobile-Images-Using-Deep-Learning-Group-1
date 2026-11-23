"""
Clinician Review Router for DermAI.

Endpoints:
- POST /images/{image_id}/review: Save or update expert clinician review and diagnosis confirmation
- GET  /images/{image_id}/review: Fetch clinician review record for an image (if exists)
- GET  /clinician/pending: List images awaiting clinician review (paginated, most recent first)
- GET  /clinician/stats: Summary metrics for clinician dashboard (cases reviewed, pending, AI accuracy, overrides)
- GET  /clinician/cases: Filterable table of cases (All, Pending, Reviewed, Overridden)
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, or_, and_, desc, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user, require_role
from app.models.case import Case
from app.models.image import Image
from app.models.patient import Patient
from app.models.prediction import Prediction
from app.models.user import User, UserRole
from app.models.clinical_review import ClinicalReview
from app.schemas.review import (
    ClinicalReviewCreate,
    ClinicalReviewResponse,
    PendingReviewItem,
    ClinicianStatsResponse,
    ClinicianCaseItem,
)
from app.services.auth_service import record_audit_log
from app.services.risk_scorer import compute_risk

router = APIRouter(tags=["Clinical Reviews"])

CLASS_MAP = {
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "bkl": "Benign Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "vasc": "Vascular Lesion",
    "df": "Dermatofibroma",
}


def _get_patient_label(c: Optional[Case]) -> str:
    if not c or not c.patient:
        return f"Patient #{c.patient_id if c else 'N/A'}"
    p = c.patient
    if p.notes and len(p.notes.strip()) > 0:
        return p.notes.strip()[:30]
    gender_str = (p.gender or "").capitalize()
    age_str = f"{p.age}y" if p.age else ""
    details = " ".join(filter(None, [gender_str, age_str]))
    return f"Patient #{p.id} ({details})" if details else f"Patient #{p.id}"


@router.post(
    "/images/{image_id}/review",
    response_model=ClinicalReviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Save or update clinician diagnostic review",
    description="Saves clinician observation notes, confirmed diagnosis class, and override flag for an image.",
)
def save_clinical_review(
    image_id: int,
    payload: ClinicalReviewCreate,
    current_user: User = Depends(require_role(UserRole.CLINICIAN, UserRole.ADMIN, "clinician", "admin")),
    db: Session = Depends(get_db),
):
    # Verify image exists
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID {image_id} not found.",
        )

    # Clean class code
    class_code = payload.confirmed_class.lower().strip()

    # Upsert review record
    review = db.query(ClinicalReview).filter(ClinicalReview.image_id == image_id).first()
    if review:
        review.clinician_id = current_user.id
        review.notes = payload.notes
        review.confirmed_class = class_code
        review.overridden = payload.overridden
        review.created_at = datetime.now(timezone.utc)
    else:
        review = ClinicalReview(
            image_id=image_id,
            clinician_id=current_user.id,
            notes=payload.notes,
            confirmed_class=class_code,
            overridden=payload.overridden,
            created_at=datetime.now(timezone.utc),
        )
        db.add(review)

    # Update parent case status to reviewed if associated
    if image.case:
        image.case.status = "reviewed"

    db.commit()
    db.refresh(review)

    # Log to audit trail
    record_audit_log(
        db=db,
        action="CLINICAL_REVIEW_SUBMITTED",
        user_id=current_user.id,
        details={
            "image_id": image_id,
            "confirmed_class": class_code,
            "overridden": payload.overridden,
            "review_id": review.id,
        },
    )

    return ClinicalReviewResponse(
        id=review.id,
        image_id=review.image_id,
        clinician_id=review.clinician_id,
        notes=review.notes,
        confirmed_class=review.confirmed_class,
        confirmed_class_name=CLASS_MAP.get(review.confirmed_class, review.confirmed_class),
        overridden=review.overridden,
        created_at=review.created_at,
        clinician_name=current_user.full_name or current_user.email,
    )


@router.get(
    "/images/{image_id}/review",
    response_model=ClinicalReviewResponse,
    summary="Fetch clinician review for an image",
    description="Retrieves the existing clinical review record for the given image if available.",
)
def get_clinical_review(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    review = db.query(ClinicalReview).filter(ClinicalReview.image_id == image_id).first()
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clinical review for image {image_id} not found.",
        )

    clinician_name = None
    if review.clinician:
        clinician_name = review.clinician.full_name or review.clinician.email

    return ClinicalReviewResponse(
        id=review.id,
        image_id=review.image_id,
        clinician_id=review.clinician_id,
        notes=review.notes,
        confirmed_class=review.confirmed_class,
        confirmed_class_name=CLASS_MAP.get(review.confirmed_class, review.confirmed_class),
        overridden=review.overridden,
        created_at=review.created_at,
        clinician_name=clinician_name,
    )


@router.get(
    "/clinician/pending",
    response_model=Union[List[PendingReviewItem], dict],
    summary="List pending images awaiting clinician review",
    description="Returns a paginated list of images that have not yet been evaluated by a clinician, most recent first.",
)
def get_pending_reviews(
    response: Response,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    envelope: bool = Query(False, description="Return { items, total } envelope"),
    current_user: User = Depends(require_role(UserRole.CLINICIAN, UserRole.ADMIN, "clinician", "admin")),
    db: Session = Depends(get_db),
):
    # Select of already reviewed image IDs
    reviewed_ids = select(ClinicalReview.image_id)

    # Images not in reviewed_ids
    query = (
        db.query(Image)
        .filter(~Image.id.in_(reviewed_ids))
        .order_by(desc(Image.id))
    )

    total = query.count()
    offset = (page - 1) * limit
    images = query.offset(offset).limit(limit).all()

    items: List[PendingReviewItem] = []
    for img in images:
        latest_pred = (
            db.query(Prediction)
            .filter(Prediction.image_id == img.id)
            .order_by(desc(Prediction.created_at), desc(Prediction.id))
            .first()
        )

        pred_code = latest_pred.predicted_class if latest_pred else None
        pred_name = CLASS_MAP.get(pred_code.lower(), pred_code) if pred_code else "Pending AI"
        conf = round(float(latest_pred.confidence), 4) if latest_pred else 0.0

        risk = None
        if latest_pred:
            risk = latest_pred.risk_level or compute_risk(pred_code, conf)["risk_level"]

        items.append(
            PendingReviewItem(
                image_id=img.id,
                case_id=img.case_id,
                patient_id=img.case.patient_id if img.case else None,
                patient_name=_get_patient_label(img.case),
                captured_at=img.captured_at,
                storage_key=img.storage_key,
                predicted_class=pred_code,
                predicted_class_name=pred_name,
                confidence=conf,
                risk_level=risk,
                top_n=latest_pred.top_n if latest_pred else [],
            )
        )

    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Page"] = str(page)
    response.headers["X-Per-Page"] = str(limit)

    if envelope:
        return {
            "items": [item.model_dump() for item in items],
            "total": total,
            "page": page,
            "limit": limit,
        }
    return items


@router.get(
    "/clinician/stats",
    response_model=ClinicianStatsResponse,
    summary="Clinician review and AI performance statistics",
    description="Aggregate metrics: total cases reviewed, pending reviews, AI accuracy vs confirmations, and weekly overrides.",
)
def get_clinician_stats(
    current_user: User = Depends(require_role(UserRole.CLINICIAN, UserRole.ADMIN, "clinician", "admin")),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    # 1. Total reviewed
    total_reviewed = db.query(func.count(ClinicalReview.id)).scalar() or 0

    # 2. Total pending images
    reviewed_ids = select(ClinicalReview.image_id)
    pending_count = (
        db.query(func.count(Image.id))
        .filter(~Image.id.in_(reviewed_ids))
        .scalar()
        or 0
    )

    # 3. Overrides this week
    overrides_this_week = (
        db.query(func.count(ClinicalReview.id))
        .filter(ClinicalReview.overridden == True, ClinicalReview.created_at >= week_ago)
        .scalar()
        or 0
    )

    # 4. AI Accuracy (% where AI matched clinician confirmation / not overridden)
    if total_reviewed > 0:
        non_overridden = (
            db.query(func.count(ClinicalReview.id))
            .filter(ClinicalReview.overridden == False)
            .scalar()
            or 0
        )
        ai_accuracy = round((non_overridden / total_reviewed) * 100, 1)
    else:
        ai_accuracy = 100.0

    return ClinicianStatsResponse(
        total_cases_reviewed=total_reviewed,
        pending_review=pending_count,
        ai_accuracy=ai_accuracy,
        overrides_this_week=overrides_this_week,
    )


@router.get(
    "/clinician/cases",
    response_model=Union[List[ClinicianCaseItem], dict],
    summary="Clinician dashboard case overview with filters",
    description="Lists cases with filter support for All, Pending, Reviewed, and Overridden.",
)
def get_clinician_cases(
    response: Response,
    filter_type: str = Query("all", alias="filter", description="Filter: all, pending, reviewed, overridden"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    envelope: bool = Query(False),
    current_user: User = Depends(require_role(UserRole.CLINICIAN, UserRole.ADMIN, "clinician", "admin")),
    db: Session = Depends(get_db),
):
    reviewed_ids = select(ClinicalReview.image_id)

    query = db.query(Image)

    filter_type_norm = filter_type.lower().strip()
    if filter_type_norm == "pending":
        query = query.filter(~Image.id.in_(reviewed_ids))
    elif filter_type_norm == "reviewed":
        query = query.filter(Image.id.in_(reviewed_ids))
    elif filter_type_norm == "overridden":
        overridden_ids = (
            select(ClinicalReview.image_id)
            .filter(ClinicalReview.overridden == True)
        )
        query = query.filter(Image.id.in_(overridden_ids))

    query = query.order_by(desc(Image.id))
    total = query.count()
    offset = (page - 1) * limit
    images = query.offset(offset).limit(limit).all()

    items: List[ClinicianCaseItem] = []
    for img in images:
        latest_pred = (
            db.query(Prediction)
            .filter(Prediction.image_id == img.id)
            .order_by(desc(Prediction.created_at), desc(Prediction.id))
            .first()
        )
        review = db.query(ClinicalReview).filter(ClinicalReview.image_id == img.id).first()

        pred_code = latest_pred.predicted_class if latest_pred else "N/A"
        pred_name = CLASS_MAP.get(pred_code.lower(), pred_code) if latest_pred else "Pending AI"
        conf = round(float(latest_pred.confidence), 4) if latest_pred else 0.0

        risk = "LOW"
        if latest_pred:
            risk = latest_pred.risk_level or compute_risk(pred_code, conf)["risk_level"]

        confirmed_code = review.confirmed_class if review else None
        confirmed_name = CLASS_MAP.get(confirmed_code, confirmed_code) if confirmed_code else None

        items.append(
            ClinicianCaseItem(
                image_id=img.id,
                case_id=img.case_id,
                patient_name=_get_patient_label(img.case),
                storage_key=img.storage_key,
                predicted_class=pred_code,
                predicted_class_name=pred_name,
                confidence=conf,
                risk_level=risk,
                date=img.captured_at,
                reviewed=review is not None,
                confirmed_class=confirmed_code,
                confirmed_class_name=confirmed_name,
                overridden=bool(review.overridden if review else False),
                notes=review.notes if review else None,
                review_id=review.id if review else None,
            )
        )

    response.headers["X-Total-Count"] = str(total)
    if envelope:
        return {
            "items": [item.model_dump() for item in items],
            "total": total,
            "page": page,
            "limit": limit,
        }
    return items
