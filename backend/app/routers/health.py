"""
Health check and system diagnostics router.
"""

from datetime import datetime, timezone
from fastapi import APIRouter
from app.database import check_database_connection
from app.services.storage_service import check_minio_connection
from app.schemas.health import HealthResponse

router = APIRouter(tags=["System Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API and infrastructure health",
    description="Returns API status, ISO datetime timestamp, and connectivity for DB and MinIO.",
)
def get_health() -> HealthResponse:
    db_connected = check_database_connection()
    minio_connected = check_minio_connection()
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        db="ok" if db_connected else "error",
        minio="ok" if minio_connected else "error",
    )
