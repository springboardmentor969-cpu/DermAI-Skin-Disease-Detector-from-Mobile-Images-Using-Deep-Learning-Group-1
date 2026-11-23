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
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import settings
from app.core.limiter import limiter
from app.routers import (
    health_router,
    auth_router,
    users_router,
    cases_router,
    images_router,
    history_router,
    dashboard_router,
    reviews_router,
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

# 1. Rate Limiting setup via slowapi
def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Too many requests — please wait before uploading again"},
    )

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# 2. Security Headers Middleware (adds nosniff, DENY, and XSS protection on every request)
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# 3. TrustedHostMiddleware for HTTP host header validation
is_production = settings.ENVIRONMENT.lower() == "production"
if is_production:
    allowed_hosts = [h.strip() for h in getattr(settings, "ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h.strip()]
    if not allowed_hosts:
        allowed_hosts = ["localhost", "127.0.0.1"]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
else:
    # Allow local and test hosts in non-production environments
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# 4. CORS configuration
raw_origins = getattr(settings, "ALLOWED_ORIGINS", "")
if isinstance(raw_origins, str):
    parsed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
elif isinstance(raw_origins, (list, tuple)):
    parsed_origins = list(raw_origins)
else:
    parsed_origins = []

if is_production:
    # Never allow wildcard "*" in production
    allowed_origins = [o for o in parsed_origins if o != "*"]
    if not allowed_origins:
        allowed_origins = ["https://dermai.example.com"]
else:
    # In development: allow localhost:5173 and localhost:3000
    dev_defaults = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    allowed_origins = list(dict.fromkeys(parsed_origins + dev_defaults))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. GZip response compression for responses >= 1000 bytes
app.add_middleware(GZipMiddleware, minimum_size=1000)



# Register routers
app.include_router(health_router)
app.include_router(auth_router, prefix="/auth")
app.include_router(users_router, prefix="/users")
app.include_router(cases_router)
app.include_router(images_router)
app.include_router(history_router)
app.include_router(dashboard_router)
app.include_router(reviews_router)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Welcome to DermAI Screening & Decision Support API",
        "docs": "/docs",
        "health": "/health",
        "disclaimer": "Educational screening tool only, not for medical diagnosis.",
    }
