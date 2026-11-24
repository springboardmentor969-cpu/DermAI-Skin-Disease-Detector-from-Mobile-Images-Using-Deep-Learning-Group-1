from datetime import datetime, timezone
from bson import ObjectId


def create_image_document(
    case_id: ObjectId,
    original_filename: str,
    file_path: str,
    mime_type: str,
    file_size: int,
    width: int,
    height: int
):
    now = datetime.now(timezone.utc)

    return {
        "case_id": case_id,
        "original_filename": original_filename,
        "file_path": file_path,
        "mime_type": mime_type,
        "file_size": file_size,
        "width": width,
        "height": height,
        "quality_status": "pending",
        "uploaded_at": now
    }