"""
Clinician Analytics and Clinical Decision-Support Dashboard Router.
Provides aggregate screening metrics, class distributions, confidence histograms,
longitudinal volume trends, and recent clinical triage cases.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, or_, and_
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models.case import Case
from app.models.image import Image
from app.models.patient import Patient
from app.models.prediction import Prediction
from app.models.user import User, UserRole
from app.schemas.dashboard import (
    DashboardStatsResponse,
    ClassDistributionItem,
    ConfidenceDistributionItem,
    RecentCaseItem,
    CasesOverTimeItem,
)
from app.services.risk_scorer import compute_risk

router = APIRouter(prefix="/dashboard", tags=["Clinician Dashboard"])

CLASS_METADATA = [
    ("mel", "Melanoma"),
    ("nv", "Melanocytic Nevus"),
    ("bkl", "Benign Keratosis"),
    ("bcc", "Basal Cell Carcinoma"),
    ("akiec", "Actinic Keratosis"),
    ("vasc", "Vascular Lesion"),
    ("df", "Dermatofibroma"),
]

CLASS_NAME_MAP = {code: name for code, name in CLASS_METADATA}


def require_clinician_or_admin(current_user: User = Depends(get_current_user)) -> User:
    """Enforces clinician or administrator role access."""
    if current_user.role not in (UserRole.CLINICIAN, UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Clinician or administrator privileges required.",
        )
    return current_user


@router.get(
    "/stats",
    response_model=DashboardStatsResponse,
    summary="Get high-level clinician dashboard metrics",
)
def get_dashboard_stats(
    current_user: User = Depends(require_clinician_or_admin),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    # 1. Total cases count
    total_cases = db.query(func.count(Case.id)).scalar() or 0

    # 2. Cases created this week (last 7 days)
    cases_this_week = (
        db.query(func.count(Case.id))
        .filter(Case.created_at >= week_ago)
        .scalar()
        or 0
    )

    # 3. Cases created this month (last 30 days)
    cases_this_month = (
        db.query(func.count(Case.id))
        .filter(Case.created_at >= month_ago)
        .scalar()
        or 0
    )

    # 4. Average prediction confidence
    avg_conf = db.query(func.avg(Prediction.confidence)).scalar()
    avg_confidence = round(float(avg_conf), 4) if avg_conf is not None else 0.0

    # 5. High-risk predictions count (HIGH risk or melanoma)
    high_risk_count = (
        db.query(func.count(Prediction.id))
        .filter(
            or_(
                func.upper(Prediction.risk_level) == "HIGH",
                and_(
                    Prediction.risk_level.is_(None),
                    func.lower(Prediction.predicted_class) == "mel",
                ),
            )
        )
        .scalar()
        or 0
    )

    # 6. Pending review cases count
    pending_review_count = (
        db.query(func.count(Case.id))
        .filter(Case.status.in_(["submitted", "reviewing"]))
        .scalar()
        or 0
    )

    return DashboardStatsResponse(
        total_cases=total_cases,
        cases_this_week=cases_this_week,
        cases_this_month=cases_this_month,
        avg_confidence=avg_confidence,
        high_risk_count=high_risk_count,
        pending_review_count=pending_review_count,
    )


@router.get(
    "/class-distribution",
    response_model=List[ClassDistributionItem],
    summary="Get distribution of AI predictions across all 7 skin lesion classes",
)
def get_class_distribution(
    current_user: User = Depends(require_clinician_or_admin),
    db: Session = Depends(get_db),
):
    counts_query = (
        db.query(func.lower(Prediction.predicted_class), func.count(Prediction.id))
        .group_by(func.lower(Prediction.predicted_class))
        .all()
    )
    counts_map = {code: count for code, count in counts_query}
    total_predictions = sum(counts_map.values())

    distribution: List[ClassDistributionItem] = []
    for code, name in CLASS_METADATA:
        cnt = counts_map.get(code, 0)
        pct = round((cnt / total_predictions * 100), 1) if total_predictions > 0 else 0.0
        distribution.append(
            ClassDistributionItem(
                class_name=name,
                class_code=code,
                count=cnt,
                percentage=pct,
            )
        )

    return distribution


@router.get(
    "/confidence-distribution",
    response_model=List[ConfidenceDistributionItem],
    summary="Get confidence histogram in 10% buckets",
)
def get_confidence_distribution(
    current_user: User = Depends(require_clinician_or_admin),
    db: Session = Depends(get_db),
):
    confidences = [c[0] for c in db.query(Prediction.confidence).all() if c[0] is not None]
    bucket_labels = [
        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]
    counts = {b: 0 for b in bucket_labels}

    for val in confidences:
        fval = float(val)
        if fval >= 1.0:
            idx = 9
        elif fval <= 0.0:
            idx = 0
        else:
            idx = min(9, int(fval * 10))
        counts[bucket_labels[idx]] += 1

    return [ConfidenceDistributionItem(bucket=b, count=counts[b]) for b in bucket_labels]


@router.get(
    "/recent-cases",
    response_model=List[RecentCaseItem],
    summary="Get last 20 clinical screening cases with latest prediction",
)
def get_recent_cases(
    current_user: User = Depends(require_clinician_or_admin),
    db: Session = Depends(get_db),
):
    cases = (
        db.query(Case)
        .order_by(Case.created_at.desc())
        .limit(20)
        .all()
    )

    items: List[RecentCaseItem] = []
    for c in cases:
        # Find latest prediction across all images in the case
        latest_pred: Optional[Prediction] = None
        if c.images:
            for img in c.images:
                if img.predictions:
                    for p in img.predictions:
                        if not latest_pred or p.created_at > latest_pred.created_at:
                            latest_pred = p

        patient_name = f"Patient #{c.patient_id}"
        if c.patient:
            if c.patient.notes and len(c.patient.notes.strip()) > 0:
                patient_name = c.patient.notes.strip()[:30]
            elif c.patient.gender or c.patient.age:
                gender_str = (c.patient.gender or "").capitalize()
                age_str = f"{c.patient.age}y" if c.patient.age else ""
                patient_name = f"Patient #{c.patient_id} ({' '.join(filter(None, [gender_str, age_str]))})"

        if latest_pred:
            pred_code = latest_pred.predicted_class
            pred_name = CLASS_NAME_MAP.get(pred_code.lower(), pred_code)
            conf = round(float(latest_pred.confidence), 4)
            risk = latest_pred.risk_level
            if not risk:
                risk = compute_risk(pred_code, conf)["risk_level"]
        else:
            pred_code = "N/A"
            pred_name = "Pending Analysis"
            conf = 0.0
            risk = "LOW"

        items.append(
            RecentCaseItem(
                case_id=c.id,
                patient_name=patient_name,
                predicted_class=pred_code,
                predicted_class_name=pred_name,
                confidence=conf,
                risk_level=risk,
                created_at=c.created_at,
            )
        )

    return items


@router.get(
    "/cases-over-time",
    response_model=List[CasesOverTimeItem],
    summary="Daily case submission count for the last 30 days",
)
def get_cases_over_time(
    current_user: User = Depends(require_clinician_or_admin),
    db: Session = Depends(get_db),
):
    today = datetime.now(timezone.utc).date()
    start_date = today - timedelta(days=29)

    # Query daily counts in last 30 days
    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    cases_grouped = (
        db.query(func.date(Case.created_at), func.count(Case.id))
        .filter(Case.created_at >= start_dt)
        .group_by(func.date(Case.created_at))
        .all()
    )

    counts_by_date = {str(d).split()[0]: count for d, count in cases_grouped}

    result: List[CasesOverTimeItem] = []
    for i in range(30):
        d_str = (start_date + timedelta(days=i)).isoformat()
        result.append(
            CasesOverTimeItem(
                date=d_str,
                count=counts_by_date.get(d_str, 0),
            )
        )

    return result
