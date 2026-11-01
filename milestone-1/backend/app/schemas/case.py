from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CaseCreate(BaseModel):
    title: str = Field(..., min_length=2, max_length=255)
    description: str | None = None
    status: str = "new"


class CaseRead(BaseModel):
    id: int
    title: str
    description: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime
    user_id: int

    class Config:
        from_attributes = True
