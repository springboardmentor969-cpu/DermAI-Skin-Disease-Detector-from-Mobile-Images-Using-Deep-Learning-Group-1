from pydantic import BaseModel, Field


class CaseCreate(BaseModel):
    title: str = Field(..., min_length=2, max_length=150)
    description: str | None = Field(
        default=None,
        max_length=1000
    )


class CaseResponse(BaseModel):
    id: str
    title: str
    description: str | None
    status: str
    clinical_note: str = ""
    created_at: str
    updated_at: str