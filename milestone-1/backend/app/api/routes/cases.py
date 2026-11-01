from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.session import get_db
from app.models.case import CaseRecord
from app.models.user import User
from app.schemas.case import CaseCreate, CaseRead
from app.services.case_service import create_case, get_case_by_id, list_cases_for_user

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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[CaseRecord]:
    return list_cases_for_user(db, current_user.id)


@router.get("/{case_id}", response_model=CaseRead)
def get_case(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CaseRecord:
    case = get_case_by_id(db, current_user.id, case_id)
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    return case
