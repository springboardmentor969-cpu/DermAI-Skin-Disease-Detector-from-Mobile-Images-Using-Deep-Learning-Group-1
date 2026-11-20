from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.database.session import get_db
from app.models.image import ImageRecord
from app.models.user import User
from app.services.case_service import get_case_by_id
from app.services.classification_service import (
    SELECTION_PATH,
    get_models_comparison_metrics,
)
from app.services.lesion_detection_service import get_lesion_detection_status
from app.services.risk_service import assess_skin_disease_risk

router = APIRouter(prefix="/analysis", tags=["analysis"])


class SessionComparisonRequest(BaseModel):
    baseline_image_id: int
    followup_image_id: int


@router.get("/status")
def get_analysis_status() -> dict[str, object]:
    return {
        "image_quality_assessment": {"status": "available", "method": "heuristic"},
        "disease_classification": {
            "status": "available" if SELECTION_PATH.is_file() else "not_trained",
            "supported_models": [
                "mobilenet_v2",
                "efficientnet_b0",
                "resnet50",
                "densenet121",
                "vit_b_16",
                "ensemble",
            ],
        },
        "explainable_ai": {
            "status": "available",
            "methods": ["Grad-CAM", "Transformer Attention Saliency"],
        },
        "risk_assessment": {
            "status": "available",
            "categories": ["Low Risk", "Moderate Risk", "High Risk", "Requires Clinical Review"],
        },
        "lesion_detection": get_lesion_detection_status(),
        "dataset_type": "HAM10000 dermoscopic images",
        "clinical_disclaimer": (
            f"{settings.PROJECT_NAME} is an AI-assisted screening decision support "
            "platform and does not provide an autonomous medical diagnosis or clinical prescription."
        ),
    }


@router.get("/models")
def get_models_comparison() -> dict[str, Any]:
    """
    Returns comparative evaluation metrics across all architectures
    as required by Milestone 3 (PDF Section 7, 10, 13).
    """
    return get_models_comparison_metrics()


@router.post("/compare")
def compare_analysis_sessions(
    payload: SessionComparisonRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Compares two analysis sessions (baseline vs follow-up) for longitudinal
    tracking as specified in PDF Section 16 & Milestone 3.
    """
    baseline = db.query(ImageRecord).filter(ImageRecord.id == payload.baseline_image_id).first()
    followup = db.query(ImageRecord).filter(ImageRecord.id == payload.followup_image_id).first()

    if not baseline or not followup:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or both images not found")

    case_baseline = get_case_by_id(db, current_user, baseline.case_id)
    case_followup = get_case_by_id(db, current_user, followup.case_id)
    if not case_baseline or not case_followup:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    base_analysis = baseline.analysis or {}
    foll_analysis = followup.analysis or {}

    base_clf = base_analysis.get("classification") or {}
    foll_clf = foll_analysis.get("classification") or {}

    base_risk = base_analysis.get("risk_assessment") or {}
    foll_risk = foll_analysis.get("risk_assessment") or {}

    # Compare predictions
    pred_match = (
        base_clf.get("predicted_class") == foll_clf.get("predicted_class")
        if base_clf.get("predicted_class") and foll_clf.get("predicted_class")
        else None
    )

    base_conf = float(base_clf.get("confidence", 0.0))
    foll_conf = float(foll_clf.get("confidence", 0.0))
    conf_delta = round(foll_conf - base_conf, 4)

    base_sev = int(base_risk.get("severity_score", 0))
    foll_sev = int(foll_risk.get("severity_score", 0))
    sev_delta = foll_sev - base_sev

    if sev_delta > 15:
        trajectory = "Risk escalation observed between sessions"
        clinical_note = "Severity score increased. Recommended scheduling prompt clinical review."
    elif sev_delta < -15:
        trajectory = "Risk de-escalation observed"
        clinical_note = "Follow-up analysis indicates reduced severity index."
    else:
        trajectory = "Stable risk presentation"
        clinical_note = "No significant divergence in risk severity detected between sessions."

    return {
        "baseline": {
            "image_id": baseline.id,
            "filename": baseline.filename,
            "date": baseline.created_at,
            "predicted_class": base_clf.get("predicted_class"),
            "confidence": base_conf,
            "risk_level": base_risk.get("risk_level", "Unknown"),
            "severity_score": base_sev,
            "architecture": base_clf.get("architecture"),
        },
        "followup": {
            "image_id": followup.id,
            "filename": followup.filename,
            "date": followup.created_at,
            "predicted_class": foll_clf.get("predicted_class"),
            "confidence": foll_conf,
            "risk_level": foll_risk.get("risk_level", "Unknown"),
            "severity_score": foll_sev,
            "architecture": foll_clf.get("architecture"),
        },
        "comparison": {
            "same_condition_predicted": pred_match,
            "confidence_delta": conf_delta,
            "severity_score_delta": sev_delta,
            "trajectory": trajectory,
            "clinical_recommendation": clinical_note,
        },
        "disclaimer": (
            "Longitudinal comparison is AI-assisted decision support and does not replace "
            "serial clinical evaluation or histopathological diagnosis."
        ),
    }
