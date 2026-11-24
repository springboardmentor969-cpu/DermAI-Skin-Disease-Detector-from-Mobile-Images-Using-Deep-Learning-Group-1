from datetime import datetime, timezone
from bson import ObjectId


def create_case_document(
    user_id: ObjectId,
    title: str,
    description: str | None = None
):
    now = datetime.now(timezone.utc)

    return {
        "user_id": user_id,
        "title": title,
        "description": description,
        "status": "pending",
        "clinical_note": "",
        "clinical_note_updated_at": None,
        "clinical_note_by": None,
        "created_at": now,
        "updated_at": now
    }