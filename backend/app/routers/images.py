"""
Skin lesion image analysis and prediction management router.

Endpoints:
- POST /images/{image_id}/analyze: Runs deep learning classification and optional segmentation.
- GET  /images/{image_id}/predictions: Lists all historical AI predictions for an image.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Response, Query
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models.image import Image
from app.models.prediction import Prediction
from app.models.user import User, UserRole
from app.schemas.case import PredictionResponse
from app.services.auth_service import record_audit_log
from app.services.storage_service import storage_service
from app.services.inference import inference_service, MANDATORY_DISCLAIMER
from app.services.risk_scorer import compute_risk

router = APIRouter(tags=["Images & AI Analysis"])


def check_image_case_access(image: Image, user: User) -> None:
    """Verifies that the current user has permission to access the image's case."""
    if not image.case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Associated clinical case not found.",
        )
    if user.role == UserRole.PATIENT and image.case.created_by != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: You do not have permission to access images for this case.",
        )


def format_prediction_response(
    pred: Prediction,
    overlay: Optional[str] = None,
    demo_model: Optional[bool] = None,
    ensemble: Optional[bool] = None,
    gradcam_key: Optional[str] = None,
    risk_level: Optional[str] = None,
    risk_color: Optional[str] = None,
    risk_message: Optional[str] = None,
    urgency_days: Optional[int] = None,
    probabilities: Optional[dict] = None,
    all_predictions: Optional[list] = None,
) -> PredictionResponse:
    """Enriches a Prediction DB record with clinical review warning, demo model flag, risk scores, and disclaimer."""
    is_low_conf = bool(pred.confidence < settings.CONFIDENCE_THRESHOLD)
    warning = "low confidence - clinical review recommended" if is_low_conf else None
    is_demo = demo_model if demo_model is not None else inference_service.is_demo_model
    is_ens = ensemble if ensemble is not None else inference_service.is_ensemble

    # Determine gradcam_key
    gk = gradcam_key
    if gk is None and not is_demo:
        cand_key = f"gradcam/{pred.image_id}.png"
        try:
            storage_service.s3_client.head_object(Bucket=storage_service.bucket_name, Key=cand_key)
            gk = cand_key
        except Exception:
            gk = None

    # Calculate 3-tier risk score if not explicitly passed
    risk_info = compute_risk(pred.predicted_class, pred.confidence)
    final_risk_level = risk_level or getattr(pred, "risk_level", None) or risk_info["risk_level"]
    final_risk_color = risk_color or risk_info["risk_color"]
    final_risk_msg = risk_message or risk_info["risk_message"]
    final_urgency_days = urgency_days or risk_info["urgency_days"]

    return PredictionResponse(
        id=pred.id,
        image_id=pred.image_id,
        model_version=pred.model_version,
        predicted_class=pred.predicted_class,
        confidence=round(float(pred.confidence), 4),
        top_n=pred.top_n or [],
        created_at=pred.created_at,
        is_low_confidence=is_low_conf,
        warning=warning,
        demo_model=is_demo,
        segmentation_overlay=overlay,
        disclaimer=MANDATORY_DISCLAIMER,
        ensemble=is_ens,
        gradcam_key=gk,
        risk_level=final_risk_level,
        risk_color=final_risk_color,
        risk_message=final_risk_msg,
        urgency_days=final_urgency_days,
        probabilities=probabilities,
        all_predictions=all_predictions,
    )


@router.post(
    "/images/{image_id}/analyze",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a skin lesion image with AI decision support",
    description=(
        "Executes AI screening on a lesion image. "
        "Refuses poor quality images, runs preprocessing and optional segmentation, "
        "persists the prediction, logs the action to audit_logs, and includes a mandatory medical disclaimer."
    ),
)
def analyze_image(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # 1. Fetch image
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID {image_id} not found.",
        )

    # 2. Check authorization
    check_image_case_access(image, current_user)

    # 3. Refuse images labelled 'poor' quality
    if (image.quality_label or "").lower() == "poor":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Cannot analyze image labelled 'poor' quality. "
                "Please retake the photo in bright, even lighting with the lesion clearly in focus "
                "before requesting AI analysis."
            ),
        )

    # 4. Fetch image bytes from storage
    try:
        image_bytes = storage_service.get_file(image.storage_key)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve image file from storage: {str(e)}",
        )

    # 5. Run inference
    try:
        analysis_result = inference_service.predict(
            image_bytes,
            run_segmentation=settings.ENABLE_SEGMENTATION,
            threshold=settings.CONFIDENCE_THRESHOLD,
            image_id=image.id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI model inference failed: {str(e)}",
        )

    confidence = float(analysis_result["confidence"])
    is_low_conf = bool(confidence < settings.CONFIDENCE_THRESHOLD)

    # 6. Save prediction in database
    prediction = Prediction(
        image_id=image.id,
        model_version=analysis_result["model_version"],
        predicted_class=analysis_result["predicted_class"],
        confidence=confidence,
        risk_level=analysis_result.get("risk_level"),
        top_n=analysis_result.get("all_predictions") or analysis_result.get("top_n", []),
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    # 7. Record audit log
    record_audit_log(
        db=db,
        action="IMAGE_ANALYZED",
        user_id=current_user.id,
        details={
            "image_id": image.id,
            "case_id": image.case_id,
            "prediction_id": prediction.id,
            "predicted_class": prediction.predicted_class,
            "confidence": prediction.confidence,
            "risk_level": prediction.risk_level,
            "is_low_confidence": is_low_conf,
            "model_version": prediction.model_version,
            "segmentation_performed": analysis_result.get("segmentation") is not None,
            "gradcam_generated": analysis_result.get("gradcam_key") is not None,
        },
    )

    return format_prediction_response(
        prediction,
        overlay=analysis_result.get("segmentation_overlay"),
        demo_model=analysis_result.get("demo_model"),
        ensemble=analysis_result.get("ensemble"),
        gradcam_key=analysis_result.get("gradcam_key"),
        risk_level=analysis_result.get("risk_level"),
        risk_color=analysis_result.get("risk_color"),
        risk_message=analysis_result.get("risk_message"),
        urgency_days=analysis_result.get("urgency_days"),
        probabilities=analysis_result.get("probabilities"),
        all_predictions=analysis_result.get("all_predictions"),
    )


@router.get(
    "/images/{image_id}/predictions",
    response_model=List[PredictionResponse],
    summary="List stored AI predictions for an image",
    description="Returns all historical predictions for the specified image, ordered newest first.",
)
def list_image_predictions(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID {image_id} not found.",
        )

    check_image_case_access(image, current_user)

    predictions = (
        db.query(Prediction)
        .filter(Prediction.image_id == image_id)
        .order_by(Prediction.created_at.desc())
        .all()
    )

    return [format_prediction_response(p) for p in predictions]


# Optional helper alias routes under /cases/{case_id}/images/{image_id}/... for frontend convenience
@router.post(
    "/cases/{case_id}/images/{image_id}/analyze",
    response_model=PredictionResponse,
    include_in_schema=False,
)
def analyze_case_image(
    case_id: int,
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return analyze_image(image_id=image_id, current_user=current_user, db=db)


@router.get(
    "/cases/{case_id}/images/{image_id}/predictions",
    response_model=List[PredictionResponse],
    include_in_schema=False,
)
def list_case_image_predictions(
    case_id: int,
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return list_image_predictions(image_id=image_id, current_user=current_user, db=db)


@router.get(
    "/images/{image_id}/gradcam",
    summary="Get Grad-CAM explainability image",
    description="Returns the Grad-CAM class activation image (pure heatmap or overlay) as image/png.",
)
def get_image_gradcam(
    image_id: int,
    type: str = Query("overlay", description="Type of Grad-CAM image: 'overlay' or 'heatmap'"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns the Grad-CAM class activation image for an analyzed image.
    Supports ?type=overlay (blended heatmap on image) or ?type=heatmap (pure jet heatmap on black).
    Requires authentication and verifies image case ownership/access.
    """
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID {image_id} not found.",
        )

    check_image_case_access(image, current_user)

    img_type = (type or "overlay").strip().lower()
    if img_type == "heatmap":
        keys_to_try = [
            f"gradcam/heatmap/{image_id}.png",
            f"gradcam/{image_id}.png",
        ]
    else:
        keys_to_try = [
            f"gradcam/overlay/{image_id}.png",
            f"gradcam/{image_id}.png",
        ]

    image_bytes = None
    for key in keys_to_try:
        try:
            image_bytes = storage_service.get_file(key)
            if image_bytes:
                break
        except Exception:
            continue

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Grad-CAM {img_type} image not found for image ID {image_id}.",
        )

    return Response(content=image_bytes, media_type="image/png")


@router.get(
    "/cases/{case_id}/images/{image_id}/gradcam",
    include_in_schema=False,
)
def get_case_image_gradcam(
    case_id: int,
    image_id: int,
    type: str = Query("overlay", description="Type of Grad-CAM image: 'overlay' or 'heatmap'"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_image_gradcam(image_id=image_id, type=type, current_user=current_user, db=db)


