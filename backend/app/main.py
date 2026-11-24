from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import test_database_connection
from routers.auth import router as auth_router
from routers.cases import router as cases_router
from routers.images import router as images_router
from routers.analysis import router as analysis_router
from routers.admin import router as admin_router
from routers.doctor import router as doctor_router
from routers.notifications import router as notifications_router

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="DermAI API",
    description="Backend API for DermAI Skin Disease Detector",
    version="1.0.0"
)


origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    test_database_connection()


# ============================================================
# ROUTERS
# ============================================================

app.include_router(
    auth_router
)
app.include_router(
    cases_router
)
app.include_router(
    images_router
)
app.include_router(
    analysis_router
)
app.include_router(
    admin_router
)
app.include_router(
    doctor_router
)
app.include_router(
    notifications_router
)

# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "DermAI",
        "message": "DermAI backend is running",
        "status": "success"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    database_status = test_database_connection()

    return {
        "status": "healthy",
        "database": (
            "connected"
            if database_status
            else "disconnected"
        )
    }