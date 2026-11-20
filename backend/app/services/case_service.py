from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.case import CaseRecord
from app.models.user import User


def create_case(db: Session, user_id: int, title: str, description: str | None = None) -> CaseRecord:
    case = CaseRecord(
        title=title,
        description=description,
        status="new",
        review_status="pending_review",
        user_id=user_id,
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def list_cases_for_user(db: Session, user: User | int, doctor_view: bool = False) -> list[CaseRecord]:
    if isinstance(user, int):
        return db.query(CaseRecord).filter(CaseRecord.user_id == user).order_by(CaseRecord.created_at.desc()).all()

    if user.role in ("doctor", "clinician", "dermatologist", "admin") and doctor_view:
        return db.query(CaseRecord).order_by(CaseRecord.created_at.desc()).all()

    return db.query(CaseRecord).filter(CaseRecord.user_id == user.id).order_by(CaseRecord.created_at.desc()).all()


def get_case_by_id(db: Session, user_or_id: User | int, case_id: int) -> CaseRecord | None:
    query = db.query(CaseRecord).filter(CaseRecord.id == case_id)
    if isinstance(user_or_id, int):
        return query.filter(CaseRecord.user_id == user_or_id).first()

    if user_or_id.role in ("doctor", "clinician", "dermatologist", "admin"):
        return query.first()

    return query.filter(CaseRecord.user_id == user_or_id.id).first()


def record_clinical_review(
    db: Session,
    case_id: int,
    doctor_id: int,
    review_status: str,
    notes: str,
) -> CaseRecord | None:
    case = db.query(CaseRecord).filter(CaseRecord.id == case_id).first()
    if not case:
        return None
    case.review_status = review_status
    case.reviewed_by_id = doctor_id
    case.doctor_review_notes = notes
    case.status = "reviewed"
    db.commit()
    db.refresh(case)
    return case
