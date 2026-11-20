from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.routes.analysis import router as analysis_router
from app.api.routes.auth import router as auth_router
from app.api.routes.cases import router as cases_router
from app.api.routes.dataset import router as dataset_router
from app.api.routes.health import router as health_router
from app.api.routes.reports import router as reports_router
from app.api.routes.upload import router as upload_router
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.database.base import Base
from app.database.session import engine
from app.models.case import CaseRecord  # noqa: F401
from app.models.image import ImageRecord  # noqa: F401
from app.models.prediction import PredictionRecord  # noqa: F401
from app.models.user import User  # noqa: F401

setup_logging()
logger = logging.getLogger("dermai.api")

Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"{method} {path} - {response.status_code} ({duration_ms:.1f}ms) [client: {client_ip}]"
        )
        return response
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(
            f"ERROR {method} {path} ({duration_ms:.1f}ms) [client: {client_ip}]: {exc}",
            exc_info=True,
        )
        raise exc


app.include_router(health_router)
app.include_router(health_router, prefix=settings.API_V1_PREFIX)
app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(cases_router, prefix=settings.API_V1_PREFIX)
app.include_router(dataset_router, prefix=settings.API_V1_PREFIX)
app.include_router(analysis_router, prefix=settings.API_V1_PREFIX)
app.include_router(reports_router, prefix=settings.API_V1_PREFIX)
app.include_router(upload_router, prefix=settings.API_V1_PREFIX)


@app.on_event("startup")
def startup_event() -> None:
    Base.metadata.create_all(bind=engine)
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "DermAI backend is running"}
