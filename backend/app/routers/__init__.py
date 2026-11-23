"""
API Routers export package.
"""

from app.routers.health import router as health_router
from app.routers.auth import router as auth_router
from app.routers.users import router as users_router
from app.routers.cases import router as cases_router
from app.routers.images import router as images_router
from app.routers.history import router as history_router
from app.routers.dashboard import router as dashboard_router
from app.routers.reviews import router as reviews_router

__all__ = [
    "health_router",
    "auth_router",
    "users_router",
    "cases_router",
    "images_router",
    "history_router",
    "dashboard_router",
    "reviews_router",
]
