from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ImageUploadRead(BaseModel):
    id: int
    case_id: int
    filename: str
    content_type: str
    file_size: int
    width: int | None = None
    height: int | None = None
    color_mode: str | None = None
    status: str
    preprocessing: dict[str, object] | None = None
    processing_error: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True
