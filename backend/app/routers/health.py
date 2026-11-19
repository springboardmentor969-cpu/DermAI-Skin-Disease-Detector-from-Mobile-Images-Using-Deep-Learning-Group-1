"""
Health check and system diagnostics router.
"""

from datetime import datetime, timezone
from fastapi import APIRouter
from app.config import settings
from app.database import check_database_connection
from app.schemas.health import HealthResponse

router = APIRouter(tags=["System Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API and infrastructure health",
    description="Returns API status, active environment, and live database connectivity.",
)
def get_health() -> HealthResponse:
    db_connected = check_database_connection()
    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(timezone.utc),
        database_connected=db_connected,
        version=settings.VERSION,
        disclaimer="DermAI is an educational & screening decision-support tool, NOT a medical diagnosis.",
    )
