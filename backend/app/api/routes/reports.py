from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.session import get_db
from app.models.image import ImageRecord
from app.models.user import User
from app.services.case_service import get_case_by_id
from app.services.report_service import (
    build_case_report_data,
    generate_case_csv,
    generate_case_json,
    generate_case_pdf,
)

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/cases/{case_id}")
def export_case_report(
    case_id: int,
    format: str = Query("json", pattern="^(pdf|csv|json)$", description="Export format: pdf, csv, or json"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """
    Automated Clinical Case Analysis Report Export (Milestone 4).
    Supports PDF, CSV, and JSON formats with full diagnostic breakdown,
    probability distributions, U-Net lesion segmentation, Grad-CAM XAI,
    and doctor review notes.
    """
    case = get_case_by_id(db, current_user, case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found or access denied",
        )

    try:
        report_data = build_case_report_data(db, current_user, case_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation error: {exc}",
        )

    fmt = format.lower()
    if fmt == "pdf":
        pdf_bytes = generate_case_pdf(report_data)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="DermAI_Clinical_Report_Case_{case_id}.pdf"',
            },
        )
    elif fmt == "csv":
        csv_str = generate_case_csv(report_data)
        return Response(
            content=csv_str,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="DermAI_Clinical_Data_Case_{case_id}.csv"',
            },
        )
    else:
        json_str = generate_case_json(report_data)
        return Response(
            content=json_str,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="DermAI_Clinical_Report_Case_{case_id}.json"',
            },
        )


@router.get("/images/{image_id}")
def export_image_report(
    image_id: int,
    format: str = Query("json", pattern="^(pdf|csv|json)$", description="Export format: pdf, csv, or json"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """
    Automated Single Image Analysis Report Export (Milestone 4).
    """
    img = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if not img:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image record not found")

    case = get_case_by_id(db, current_user, img.case_id)
    if not case:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return export_case_report(
        case_id=img.case_id,
        format=format,
        current_user=current_user,
        db=db,
    )
