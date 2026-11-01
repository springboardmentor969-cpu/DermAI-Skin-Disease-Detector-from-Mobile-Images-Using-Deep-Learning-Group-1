from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
import uuid
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.image import ImageRecord

MIME_TO_FORMAT = {"image/jpeg": "JPEG", "image/png": "PNG"}
FORMAT_TO_EXTENSION = {"JPEG": ".jpg", "PNG": ".png"}


def safe_image_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def validate_image(file_bytes: bytes, content_type: str) -> tuple[int, int, str, str]:
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")
    if content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise ValueError("Only JPEG and PNG images are supported.")

    try:
        with Image.open(io.BytesIO(file_bytes)) as image:
            image_format = image.format
            width, height = image.size
            color_mode = image.mode
            if image_format not in FORMAT_TO_EXTENSION:
                raise ValueError("Only JPEG and PNG image contents are supported.")
            if MIME_TO_FORMAT[content_type] != image_format:
                raise ValueError("The file content does not match its declared image type.")
            if (
                width <= 0
                or height <= 0
                or width > settings.MAX_IMAGE_DIMENSION
                or height > settings.MAX_IMAGE_DIMENSION
            ):
                raise ValueError(
                    f"Image dimensions must not exceed "
                    f"{settings.MAX_IMAGE_DIMENSION} x {settings.MAX_IMAGE_DIMENSION} pixels."
                )
            image.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("Corrupted or unreadable image file.") from exc

    return width, height, color_mode, image_format


def _safe_original_filename(filename: str) -> str:
    basename = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", basename).strip("._")
    return cleaned[:100] or "image"


def save_upload(
    db: Session,
    case_id: int,
    filename: str,
    content_type: str,
    file_bytes: bytes,
) -> ImageRecord:
    if len(file_bytes) > settings.MAX_UPLOAD_SIZE:
        raise ValueError(
            f"Image exceeds the {settings.MAX_UPLOAD_SIZE // (1024 * 1024)} MB upload limit."
        )
    width, height, color_mode, image_format = validate_image(file_bytes, content_type)

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_original_filename(filename)
    stored_name = f"{uuid.uuid4().hex}{FORMAT_TO_EXTENSION[image_format]}"
    storage_path = (upload_dir / stored_name).resolve()
    if storage_path.parent != upload_dir.resolve():
        raise ValueError("Invalid upload storage path.")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=upload_dir,
            prefix=".upload-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(file_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        image_record = ImageRecord(
            case_id=case_id,
            filename=safe_name,
            content_type=content_type,
            storage_path=str(storage_path),
            file_size=len(file_bytes),
            width=width,
            height=height,
            color_mode=color_mode,
            image_hash=safe_image_hash(file_bytes),
            status="uploaded",
        )
        db.add(image_record)
        db.flush()
        os.replace(temporary_path, storage_path)
        temporary_path = None
        db.commit()
        return image_record
    except Exception:
        try:
            db.rollback()
        finally:
            storage_path.unlink(missing_ok=True)
        raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def get_case_images(db: Session, case_id: int) -> list[ImageRecord]:
    return db.query(ImageRecord).filter(ImageRecord.case_id == case_id).order_by(ImageRecord.created_at.desc()).all()
