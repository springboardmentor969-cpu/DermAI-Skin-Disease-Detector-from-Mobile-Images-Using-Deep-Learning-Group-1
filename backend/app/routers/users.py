"""
User profile management router.
"""

from fastapi import APIRouter, Depends
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.user import UserResponse

router = APIRouter(tags=["Users"])


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current authenticated user profile",
    description="Returns the profile information of the currently authenticated user.",
)
def get_user_me(current_user: User = Depends(get_current_user)) -> User:
    return current_user
