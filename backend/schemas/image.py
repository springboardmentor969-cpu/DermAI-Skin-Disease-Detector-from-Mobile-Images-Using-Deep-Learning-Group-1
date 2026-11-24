from pydantic import BaseModel


class ImageResponse(BaseModel):
    id: str
    case_id: str
    original_filename: str
    mime_type: str
    file_size: int
    width: int
    height: int
    quality_status: str
    uploaded_at: str