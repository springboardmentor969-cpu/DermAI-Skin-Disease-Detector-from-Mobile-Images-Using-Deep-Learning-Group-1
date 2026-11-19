"""
FastAPI route dependencies for authentication and role-based access control.
"""

from typing import Optional, Union
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User, UserRole
from app.services.auth_service import decode_access_token

# OAuth2 scheme configured to use /auth/login for Swagger UI authorization
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Validate the Bearer token, extract the user email, and retrieve the user from the database.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    email: Optional[str] = payload.get("sub")
    if email is None:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception

    return user


def require_role(*allowed_roles: Union[UserRole, str]):
    """
    Reusable dependency to protect endpoints by role.

    Usage:
        @router.get("/admin-only", dependencies=[Depends(require_role(UserRole.ADMIN))])
        def admin_view(): ...

        @router.post("/cases", response_model=...)
        def create_case(user: User = Depends(require_role("clinician", "admin"))): ...
    """
    allowed_str_roles = [
        r.value if isinstance(r, UserRole) else str(r) for r in allowed_roles
    ]

    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        user_role_str = (
            current_user.role.value
            if isinstance(current_user.role, UserRole)
            else str(current_user.role)
        )
        if user_role_str not in allowed_str_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: Insufficient permissions. Required role: {', '.join(allowed_str_roles)}",
            )
        return current_user

    return role_checker
