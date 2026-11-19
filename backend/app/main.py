"""
Main FastAPI application entrypoint for DermAI.
Initializes the application, CORS middleware, and includes API routers.
"""

import sys
from pathlib import Path

# Ensure backend directory is in sys.path when running from workspace root
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import (
    health_router,
    auth_router,
    users_router,
    cases_router,
    images_router,
    history_router,
    dashboard_router,
)
from app.services.inference import inference_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load AI models once at application startup."""
    try:
        inference_service.load_model()
    except Exception as e:
        import logging
        logging.getLogger("dermai").warning(f"Startup model load notice: {e}")
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "**DermAI API**: Automated mobile skin condition screening and decision-support tool.\n\n"
        "⚠️ **Disclaimer**: DermAI is an educational research and clinical screening decision-support aid. "
        "It does NOT provide definitive medical diagnosis."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS) for React + Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(auth_router, prefix="/auth")
app.include_router(users_router, prefix="/users")
app.include_router(cases_router)
app.include_router(images_router)
app.include_router(history_router)
app.include_router(dashboard_router)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Welcome to DermAI Screening & Decision Support API",
        "docs": "/docs",
        "health": "/health",
        "disclaimer": "Educational screening tool only, not for medical diagnosis.",
    }
