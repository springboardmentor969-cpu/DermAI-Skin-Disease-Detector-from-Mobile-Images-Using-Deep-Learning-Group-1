"""
Pydantic schema for the /health diagnostic endpoint.
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    timestamp: str
    db: str
    minio: str
