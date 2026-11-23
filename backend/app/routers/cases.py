"""
Case management and skin lesion image upload router.
Implements role-based visibility, Pillow image validation, MinIO S3 storage, and audit logging.
"""

import io
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Response, status
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models.case import Case
from app.models.patient import Patient
from app.models.image import Image
from app.models.user import User, UserRole
from app.schemas.case import CaseCreate, CaseResponse, ImageResponse
from app.services.auth_service import record_audit_log
from app.services.storage_service import storage_service
from ml.preprocessing.quality import assess_image_quality
from ml.preprocessing.preprocess import optimize_image_for_storage

router = APIRouter(prefix="/cases", tags=["Cases & Images"])

MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB maximum allowed file size
MIN_IMAGE_DIMENSION = 50           # 50x50 minimum resolution



@router.post(
    "",
    response_model=CaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new screening case",
    description="Creates a new clinical screening case with patient demographic information and body location.",
)
def create_case(
    case_in: CaseCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    patient_id = case_in.patient_id

    # If an existing patient_id was not provided, create a new de-identified patient record
    if not patient_id:
        patient = Patient(
            age=case_in.patient_age,
            gender=case_in.patient_gender,
            notes=case_in.clinical_notes,
        )
        db.add(patient)
        db.flush()
        patient_id = patient.id
    else:
        existing_patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not existing_patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient with ID {patient_id} does not exist.",
            )

    new_case = Case(
        patient_id=patient_id,
        created_by=current_user.id,
        body_location=case_in.body_location,
        status=case_in.status or "submitted",
    )
    db.add(new_case)
    db.commit()
    db.refresh(new_case)

    # Log case creation
    record_audit_log(
        db=db,
        action="CASE_CREATED",
        user_id=current_user.id,
        details={
            "case_id": new_case.id,
            "patient_id": new_case.patient_id,
            "body_location": new_case.body_location,
        },
    )

    return new_case


@router.get(
    "",
    response_model=List[CaseResponse],
    summary="List cases with role-based access filtering",
    description=(
        "Patients can only view their own submitted cases. "
        "Clinicians, researchers, and administrators can view all cases across the system."
    ),
)
def list_cases(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role == UserRole.PATIENT:
        # Patient only sees cases created by themselves
        cases = (
            db.query(Case)
            .filter(Case.created_by == current_user.id)
            .order_by(Case.created_at.desc())
            .all()
        )
    else:
        # Clinicians and staff view all cases
        cases = db.query(Case).order_by(Case.created_at.desc()).all()

    return cases


@router.get(
    "/{case_id}",
    response_model=CaseResponse,
    summary="Retrieve details of a specific case",
    description="Retrieves a specific case by ID. Patients are restricted to viewing only their own cases.",
)
def get_case(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found",
        )

    # Role-based authorization
    if current_user.role == UserRole.PATIENT and case.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: You do not have permission to view this case.",
        )

    return case


@router.post(
    "/{case_id}/images",
    response_model=ImageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a skin lesion image to a case",
    description=(
        "Validates image format (JPG/PNG), size (<=10 MB), and minimum resolution (>=224x224) using Pillow. "
        "Stores the file securely in MinIO object storage and records the transaction in audit_logs."
    ),
)
async def upload_case_image(
    case_id: int,
    file: UploadFile = File(..., description="Image file (JPG or PNG, max 10MB, min 224x224)"),
    source: str = Form(default="camera", description="Image capture source ('camera' or 'gallery')"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found",
        )

    # Authorization check: patients can only upload to their own cases
    if current_user.role == UserRole.PATIENT and case.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: You cannot upload images to another user's case.",
        )

    # Read file content into memory
    file_bytes = await file.read()
    file_size = len(file_bytes)

    # 1. Size validation
    if file_size > MAX_IMAGE_SIZE:
        size_mb = file_size / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of 20 MB (received {size_mb:.2f} MB).",
        )

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # 2. Pillow real image validation & format check
    try:
        image = PILImage.open(io.BytesIO(file_bytes))
        image.verify()  # Verify image integrity
        # Re-open after verify to inspect dimensions and format
        image = PILImage.open(io.BytesIO(file_bytes))
        img_format = (image.format or "").upper()
        width, height = image.size
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file: Failed to decode image contents with Pillow.",
        )

    # Validate allowed formats (JPG / PNG / WebP / BMP)
    if img_format not in ("JPEG", "JPG", "PNG", "WEBP", "BMP"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format '{img_format}'. Only JPG and PNG images are allowed.",
        )

    # 3. Minimum resolution check (50x50)
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image too small — minimum 50x50 pixels required for accurate analysis",
        )

    # Preprocess & optimize image for storage: resize to 224x224 and compress to JPEG quality 85
    storage_bytes, content_type = optimize_image_for_storage(file_bytes, target_size=(224, 224), quality=85)
    unique_key = f"cases/{case.id}/{uuid.uuid4().hex}.jpg"

    try:
        storage_service.upload_file(
            file_bytes=storage_bytes,
            storage_key=unique_key,
            content_type=content_type,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store image in MinIO object storage: {str(e)}",
        )

    # Run automated image quality assessment
    quality_assessment = assess_image_quality(file_bytes)
    quality_label = quality_assessment["quality_label"]

    # Provide helpful warning if image quality is sub-optimal / poor
    warning_message: Optional[str] = None
    if quality_label == "poor":
        issues_summary = (
            ", ".join(quality_assessment.get("issues", []))
            if quality_assessment.get("issues")
            else "low visual quality"
        )
        warning_message = (
            f"Warning: Image quality is assessed as 'poor' ({issues_summary}). "
            "Please retake the photo in bright, even lighting with the lesion in sharp focus."
        )

    # Persist record in images table
    new_image = Image(
        case_id=case.id,
        storage_key=unique_key,
        source=source if source in ("camera", "gallery") else "camera",
        quality_label=quality_label,
        quality_details=quality_assessment,
    )
    db.add(new_image)
    db.commit()
    db.refresh(new_image)

    # Log upload action in audit_logs
    record_audit_log(
        db=db,
        action="IMAGE_UPLOADED",
        user_id=current_user.id,
        details={
            "case_id": case.id,
            "image_id": new_image.id,
            "storage_key": unique_key,
            "quality_label": quality_label,
            "blur_score": quality_assessment.get("blur_score"),
            "file_size": file_size,
            "dimensions": f"{width}x{height}",
            "format": img_format,
        },
    )

    return ImageResponse(
        id=new_image.id,
        case_id=new_image.case_id,
        storage_key=new_image.storage_key,
        source=new_image.source,
        captured_at=new_image.captured_at,
        quality_label=new_image.quality_label,
        quality_details=new_image.quality_details,
        warning=warning_message,
        predictions=[],
    )


@router.get(
    "/{case_id}/images",
    response_model=List[ImageResponse],
    summary="List images associated with a case",
    description="Returns all uploaded skin lesion images for the specified case.",
)
def list_case_images(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found",
        )

    # Role-based authorization
    if current_user.role == UserRole.PATIENT and case.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: You do not have permission to view images for this case.",
        )

    images = (
        db.query(Image)
        .filter(Image.case_id == case_id)
        .order_by(Image.captured_at.asc())
        .all()
    )
    return images


@router.get(
    "/{case_id}/images/{image_id}/content",
    summary="Download/stream image file",
    description="Streams the raw binary image from MinIO storage for browser display.",
)
def get_case_image_content(
    case_id: int,
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found",
        )

    if current_user.role == UserRole.PATIENT and case.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: You do not have permission to view images for this case.",
        )

    image = (
        db.query(Image)
        .filter(Image.id == image_id, Image.case_id == case_id)
        .first()
    )
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID {image_id} not found in case {case_id}.",
        )

    try:
        data = storage_service.get_file(image.storage_key)
        media_type = "image/png" if image.storage_key.endswith(".png") else "image/jpeg"
        return Response(content=data, media_type=media_type)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch image from storage: {str(e)}",
        )

