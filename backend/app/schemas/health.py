"""
Pydantic schema for the /health diagnostic endpoint.
"""

from datetime import datetime
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    environment: str
    timestamp: datetime
    database_connected: bool
    version: str
    disclaimer: str
