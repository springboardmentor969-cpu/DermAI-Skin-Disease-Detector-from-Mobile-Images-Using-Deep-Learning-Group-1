from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_current_user
from app.core.config import settings
from app.database.session import get_db
from app.models.image import ImageRecord
from app.models.prediction import PredictionRecord
from app.models.user import User
from app.schemas.image import ImageUploadRead
from app.services.case_service import get_case_by_id
from app.services.classification_service import (
    HAM10000_CLASSES,
    get_model_for_explainability,
    predict_skin_disease,
)
from app.services.explainability_service import generate_gradcam_explanation
from app.services.image_quality_service import assess_image_quality
from app.services.image_service import get_case_images, save_upload
from app.services.lesion_detection_service import (
    get_lesion_detection_status,
    predict_lesion_mask,
)
from app.services.preprocessing_service import ImagePreprocessor
from app.services.risk_service import assess_skin_disease_risk

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/{case_id}", response_model=ImageUploadRead, status_code=status.HTTP_201_CREATED)
async def upload_image(
    case_id: int,
    file: UploadFile = File(...),
    architecture: str | None = Query(None, description="Model architecture to use for classification"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    case = get_case_by_id(db, current_user.id, case_id)
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    content = bytearray()
    preprocessing_metadata: dict[str, object] | None = None
    processing_error: str | None = None
    quality_assessment: dict[str, object] | None = None
    classification: dict[str, object] | None = None
    lesion_detection = get_lesion_detection_status()
    explainability: dict[str, object] | None = None
    risk_assessment: dict[str, object] | None = None
    analysis_errors: list[str] = []

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

        # 1. Image Quality Assessment
        try:
            quality_assessment = await run_in_threadpool(
                assess_image_quality,
                bytes(content),
            )
        except (ValueError, OSError, RuntimeError) as exc:
            analysis_errors.append(f"Image quality assessment failed: {exc}")

        # 2. Disease Classification
        try:
            classification = await run_in_threadpool(
                predict_skin_disease,
                bytes(content),
                architecture,
            )
        except (ValueError, OSError, RuntimeError) as exc:
            analysis_errors.append(f"Disease classification failed: {exc}")

        # 3. Lesion Segmentation
        try:
            lesion_detection = await run_in_threadpool(
                predict_lesion_mask,
                bytes(content),
            )
        except (ValueError, OSError, RuntimeError) as exc:
            analysis_errors.append(f"Lesion segmentation failed: {exc}")

        # 4. Explainable AI (Grad-CAM / Attention Visualization)
        try:
            target_arch = classification.get("architecture") if classification else architecture
            model, used_arch = get_model_for_explainability(target_arch)
            target_class_idx = None
            if classification and classification.get("predicted_class"):
                pred_cls = classification["predicted_class"]
                if pred_cls in HAM10000_CLASSES:
                    target_class_idx = HAM10000_CLASSES.index(pred_cls)

            explainability = await run_in_threadpool(
                generate_gradcam_explanation,
                model,
                used_arch,
                bytes(content),
                target_class_idx,
            )
        except Exception as exc:
            analysis_errors.append(f"Explainability generation failed: {exc}")

        # 5. Severity & Risk Assessment
        try:
            pred_class = classification.get("predicted_class") if classification else None
            conf_val = classification.get("confidence") if classification else None
            q_score = (
                quality_assessment.get("quality_score")
                if quality_assessment and "quality_score" in quality_assessment
                else None
            )
            is_poor = (
                quality_assessment.get("quality_category") == "poor"
                if quality_assessment
                else False
            )
            risk_assessment = assess_skin_disease_risk(
                pred_class,
                conf_val,
                quality_score=q_score,
                is_poor_quality=is_poor,
            )
        except Exception as exc:
            analysis_errors.append(f"Risk assessment failed: {exc}")

        # 6. Image Preprocessing
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
                {
                    "preprocessing": preprocessing_metadata,
                    "quality_assessment": quality_assessment,
                    "classification": classification,
                    "lesion_detection": lesion_detection,
                    "explainability": explainability,
                    "risk_assessment": risk_assessment,
                    "analysis_errors": analysis_errors,
                },
                separators=(",", ":"),
            )
        except (ValueError, OSError, RuntimeError) as exc:
            image_record.status = "processing_failed"
            processing_error = "Image preprocessing failed; the original upload was preserved."
            image_record.metadata_json = json.dumps(
                {
                    "processing_error": processing_error,
                    "quality_assessment": quality_assessment,
                    "classification": classification,
                    "lesion_detection": lesion_detection,
                    "explainability": explainability,
                    "risk_assessment": risk_assessment,
                    "analysis_errors": analysis_errors,
                },
                separators=(",", ":"),
            )

        # 7. Persist PredictionRecord
        if classification and classification.get("status") == "available":
            prediction_record = PredictionRecord(
                case_id=case_id,
                image_id=image_record.id,
                model_version=classification.get("architecture", "selected"),
                prediction_label=classification.get("predicted_class"),
                confidence=classification.get("confidence"),
                risk_level=risk_assessment.get("risk_level") if risk_assessment else None,
                status="completed",
            )
            db.add(prediction_record)

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
        "quality_assessment": quality_assessment,
        "classification": classification,
        "lesion_detection": lesion_detection,
        "explainability": explainability,
        "risk_assessment": risk_assessment,
        "analysis_errors": analysis_errors,
        "processing_error": processing_error,
        "created_at": image_record.created_at,
    }


@router.get("/images/{image_id}/file")
def get_image_file(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    image_record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if image_record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    case = get_case_by_id(db, current_user, image_record.case_id)
    if case is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    file_path = Path(image_record.storage_path)
    if not file_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file missing on server")

    return FileResponse(file_path, media_type=image_record.content_type)


@router.get("/cases/{case_id}/images")
def get_case_images_endpoint(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    case = get_case_by_id(db, current_user, case_id)
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    images = get_case_images(db, case_id)
    return [
        {
            "id": img.id,
            "case_id": img.case_id,
            "filename": img.filename,
            "content_type": img.content_type,
            "file_size": img.file_size,
            "width": img.width,
            "height": img.height,
            "color_mode": img.color_mode,
            "status": img.status,
            "created_at": img.created_at,
            "analysis": img.analysis,
        }
        for img in images
    ]
