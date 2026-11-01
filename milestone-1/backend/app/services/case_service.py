from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.case import CaseRecord


def create_case(db: Session, user_id: int, title: str, description: str | None = None) -> CaseRecord:
    case = CaseRecord(title=title, description=description, status="new", user_id=user_id)
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def list_cases_for_user(db: Session, user_id: int) -> list[CaseRecord]:
    return db.query(CaseRecord).filter(CaseRecord.user_id == user_id).order_by(CaseRecord.created_at.desc()).all()


def get_case_by_id(db: Session, user_id: int, case_id: int) -> CaseRecord | None:
    return (
        db.query(CaseRecord)
        .filter(CaseRecord.id == case_id, CaseRecord.user_id == user_id)
        .first()
    )
