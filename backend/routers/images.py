from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.database import cases_collection, images_collection
from app.dependencies.auth import get_current_user
from models.image import create_image_document
from schemas.image import ImageResponse


router = APIRouter(
    prefix="/cases",
    tags=["Image Upload"]
)


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

UPLOAD_DIR = PROJECT_ROOT / "uploads" / "images"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ALLOWED IMAGE TYPES
# ============================================================

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}

MAX_FILE_SIZE = 10 * 1024 * 1024


# ============================================================
# IMAGE UPDATE SCHEMA
# ============================================================

class ImageUpdate(BaseModel):

    quality_status: str = Field(
        ...,
        min_length=1,
        max_length=50
    )


# ============================================================
# HELPER — GET CASE BELONGING TO CURRENT USER
# ============================================================

def get_owned_case(
    case_id: str,
    user_id: ObjectId
):
    try:
        case_object_id = ObjectId(case_id)

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid case ID."
        )

    case = cases_collection.find_one({
        "_id": case_object_id,
        "user_id": user_id
    })

    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found."
        )

    return case


# ============================================================
# HELPER — GET IMAGE BELONGING TO CASE
# ============================================================

def get_owned_image(
    case_id: str,
    image_id: str,
    user_id: ObjectId
):
    case = get_owned_case(
        case_id=case_id,
        user_id=user_id
    )

    try:
        image_object_id = ObjectId(image_id)

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image ID."
        )

    image = images_collection.find_one({
        "_id": image_object_id,
        "case_id": case["_id"]
    })

    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found."
        )

    return case, image


# ============================================================
# HELPER — CONVERT IMAGE DOCUMENT TO RESPONSE
# ============================================================

def image_to_response(image):

    return ImageResponse(
        id=str(image["_id"]),
        case_id=str(image["case_id"]),
        original_filename=image["original_filename"],
        mime_type=image["mime_type"],
        file_size=image["file_size"],
        width=image["width"],
        height=image["height"],
        quality_status=image["quality_status"],
        uploaded_at=image["uploaded_at"].isoformat()
    )


# ============================================================
# UPLOAD IMAGE
# ============================================================

@router.post(
    "/{case_id}/images",
    response_model=ImageResponse,
    status_code=status.HTTP_201_CREATED
)
def upload_image(
    case_id: str,
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):

    case = get_owned_case(
        case_id=case_id,
        user_id=current_user["_id"]
    )

    case_object_id = case["_id"]

    # --------------------------------------------------------
    # Validate MIME type
    # --------------------------------------------------------

    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG, PNG, and WEBP images are allowed."
        )

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    file_data = file.file.read()

    file_size = len(file_data)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image size must not exceed 10 MB."
        )

    # --------------------------------------------------------
    # Validate actual image
    # --------------------------------------------------------

    try:

        image = Image.open(file.file)

        image.verify()

        file.file.seek(0)

        image = Image.open(file.file)

        width, height = image.size

    except (UnidentifiedImageError, OSError):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid readable image."
        )

    finally:

        file.file.seek(0)

    # --------------------------------------------------------
    # Generate server-side filename
    # --------------------------------------------------------

    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if not extension:
        extension = ".jpg"

    generated_filename = (
        f"{uuid4().hex}{extension}"
    )

    saved_file_path = (
        UPLOAD_DIR / generated_filename
    )

    # --------------------------------------------------------
    # Save physical image
    # --------------------------------------------------------

    with open(saved_file_path, "wb") as output_file:

        output_file.write(file_data)

    # --------------------------------------------------------
    # MongoDB document
    # --------------------------------------------------------

    image_document = create_image_document(
        case_id=case_object_id,
        original_filename=file.filename or "unknown",
        file_path=str(saved_file_path),
        mime_type=file.content_type,
        file_size=file_size,
        width=width,
        height=height
    )

    try:

        result = images_collection.insert_one(
            image_document
        )

    except Exception:

        if saved_file_path.exists():
            saved_file_path.unlink()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save image information."
        )

    image_document["_id"] = result.inserted_id

    return image_to_response(image_document)


# ============================================================
# GET ALL IMAGES OF A CASE
# ============================================================

@router.get(
    "/{case_id}/images",
    response_model=list[ImageResponse]
)
def get_case_images(
    case_id: str,
    current_user=Depends(get_current_user)
):

    case = get_owned_case(
        case_id=case_id,
        user_id=current_user["_id"]
    )

    images = images_collection.find({
        "case_id": case["_id"]
    }).sort(
        "uploaded_at",
        -1
    )

    return [
        image_to_response(image)
        for image in images
    ]


# ============================================================
# GET IMAGE METADATA
# ============================================================

@router.get(
    "/{case_id}/images/{image_id}",
    response_model=ImageResponse
)
def get_image(
    case_id: str,
    image_id: str,
    current_user=Depends(get_current_user)
):

    _, image = get_owned_image(
        case_id=case_id,
        image_id=image_id,
        user_id=current_user["_id"]
    )

    return image_to_response(image)


# ============================================================
# GET ACTUAL IMAGE FILE
# ============================================================

@router.get(
    "/{case_id}/images/{image_id}/file"
)
def get_image_file(
    case_id: str,
    image_id: str,
    current_user=Depends(get_current_user)
):

    _, image = get_owned_image(
        case_id=case_id,
        image_id=image_id,
        user_id=current_user["_id"]
    )

    file_path = Path(
        image["file_path"]
    )

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image file not found on server."
        )

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored image path is invalid."
        )

    return FileResponse(
        path=str(file_path),
        media_type=image.get(
            "mime_type",
            "application/octet-stream"
        ),
        filename=image.get(
            "original_filename",
            file_path.name
        )
    )


# ============================================================
# UPDATE IMAGE METADATA
# ============================================================

@router.put(
    "/{case_id}/images/{image_id}",
    response_model=ImageResponse
)
def update_image(
    case_id: str,
    image_id: str,
    image_data: ImageUpdate,
    current_user=Depends(get_current_user)
):

    _, image = get_owned_image(
        case_id=case_id,
        image_id=image_id,
        user_id=current_user["_id"]
    )

    updated_at = datetime.now(timezone.utc)

    images_collection.update_one(
        {"_id": image["_id"]},
        {
            "$set": {
                "quality_status": image_data.quality_status,
                "updated_at": updated_at
            }
        }
    )

    updated_image = images_collection.find_one({
        "_id": image["_id"]
    })

    return image_to_response(updated_image)


# ============================================================
# DELETE IMAGE
# ============================================================

@router.delete(
    "/{case_id}/images/{image_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
def delete_image(
    case_id: str,
    image_id: str,
    current_user=Depends(get_current_user)
):

    _, image = get_owned_image(
        case_id=case_id,
        image_id=image_id,
        user_id=current_user["_id"]
    )

    file_path = Path(
        image["file_path"]
    )

    if file_path.exists():

        try:
            file_path.unlink()

        except OSError:
            pass

    images_collection.delete_one({
        "_id": image["_id"]
    })

    return None