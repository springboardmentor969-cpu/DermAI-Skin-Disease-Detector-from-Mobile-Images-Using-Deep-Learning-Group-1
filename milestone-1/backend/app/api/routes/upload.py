from __future__ import annotations

import json

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_current_user
from app.core.config import settings
from app.database.session import get_db
from app.models.user import User
from app.schemas.image import ImageUploadRead
from app.services.case_service import get_case_by_id
from app.services.image_service import save_upload
from app.services.preprocessing_service import ImagePreprocessor

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/{case_id}", response_model=ImageUploadRead, status_code=status.HTTP_201_CREATED)
async def upload_image(
    case_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    case = get_case_by_id(db, current_user.id, case_id)
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    content = bytearray()
    preprocessing_metadata: dict[str, object] | None = None
    processing_error: str | None = None
    try:
        while chunk := await file.read(64 * 1024):
            content.extend(chunk)
            if len(content) > settings.MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"Image exceeds the "
                        f"{settings.MAX_UPLOAD_SIZE // (1024 * 1024)} MB upload limit."
                    ),
                )
        try:
            image_record = save_upload(
                db,
                case_id,
                file.filename or "image",
                file.content_type or "",
                bytes(content),
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        image_record.status = "processing"
        db.add(image_record)
        db.commit()
        try:
            processed_image, preprocessing_metadata = await run_in_threadpool(
                ImagePreprocessor().preprocess,
                bytes(content),
            )
            if (
                processed_image.ndim != 3
                or tuple(processed_image.shape[:2]) != settings.TARGET_IMAGE_SIZE
                or processed_image.shape[2] != 3
                or not np.isfinite(processed_image).all()
                or processed_image.min() < 0
                or processed_image.max() > 1
            ):
                raise ValueError("Preprocessor returned an invalid normalized RGB image.")
            preprocessing_metadata = {
                **preprocessing_metadata,
                "pixel_value_range": [
                    float(processed_image.min()),
                    float(processed_image.max()),
                ],
            }
            image_record.status = "processed"
            image_record.metadata_json = json.dumps(
                {"preprocessing": preprocessing_metadata},
                separators=(",", ":"),
            )
        except (ValueError, OSError, RuntimeError) as exc:
            image_record.status = "processing_failed"
            processing_error = "Image preprocessing failed; the original upload was preserved."
            image_record.metadata_json = json.dumps(
                {"processing_error": processing_error},
                separators=(",", ":"),
            )
        db.add(image_record)
        db.commit()
    finally:
        await file.close()

    return {
        "id": image_record.id,
        "case_id": image_record.case_id,
        "filename": image_record.filename,
        "content_type": image_record.content_type,
        "file_size": image_record.file_size,
        "width": image_record.width,
        "height": image_record.height,
        "color_mode": image_record.color_mode,
        "status": image_record.status,
        "preprocessing": preprocessing_metadata,
        "processing_error": processing_error,
        "created_at": image_record.created_at,
    }
