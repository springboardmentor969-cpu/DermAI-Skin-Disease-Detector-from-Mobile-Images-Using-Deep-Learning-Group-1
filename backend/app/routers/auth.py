"""
Authentication router: user registration and JWT token login.
Logs all events (registration, login success, login failure) to the audit_logs table.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User, UserRole
from app.schemas.user import UserCreate, UserResponse, Token
from app.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    record_audit_log,
)

router = APIRouter(tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    description=(
        "Registers a new user account with role. Note: Public registration as 'admin' is strictly prohibited."
    ),
)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    # Block attempts to register as administrator through public endpoint
    role_str = (
        user_in.role.value if isinstance(user_in.role, UserRole) else str(user_in.role)
    ).lower()

    if role_str == UserRole.ADMIN.value:
        # Audit the unauthorized registration attempt
        record_audit_log(
            db=db,
            action="USER_REGISTER_BLOCKED",
            user_id=None,
            details={
                "attempted_email": str(user_in.email),
                "attempted_role": role_str,
                "reason": "admin_registration_prohibited",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Registration as administrator is not permitted through public registration.",
        )

    # Check for existing email
    existing_user = db.query(User).filter(User.email == user_in.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already registered.",
        )

    # Hash the password using bcrypt
    hashed_password = hash_password(user_in.password)

    new_user = User(
        email=user_in.email,
        password_hash=hashed_password,
        full_name=user_in.full_name,
        role=user_in.role,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Record registration in audit_logs
    record_audit_log(
        db=db,
        action="USER_REGISTER",
        user_id=new_user.id,
        details={
            "email": new_user.email,
            "role": role_str,
            "full_name": new_user.full_name,
        },
    )

    return new_user


@router.post(
    "/login",
    response_model=Token,
    summary="Authenticate and receive JWT access token",
    description=(
        "Authenticates a user via JSON body ({email, password}) or form data ({username, password}) "
        "and returns a signed Bearer JWT token. Compatible with Swagger UI 'Authorize' button."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string", "example": "clinician@example.com"},
                            "password": {"type": "string", "example": "SecurePass123!"},
                        },
                        "required": ["email", "password"],
                    }
                },
                "application/x-www-form-urlencoded": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "username": {"type": "string", "example": "clinician@example.com"},
                            "password": {"type": "string", "example": "SecurePass123!"},
                        },
                        "required": ["username", "password"],
                    }
                },
            }
        }
    },
)
async def login(request: Request, db: Session = Depends(get_db)):
    content_type = request.headers.get("content-type", "")

    email: Optional[str] = None
    password: Optional[str] = None

    if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        form = await request.form()
        email = form.get("username") or form.get("email")
        password = form.get("password")
    else:
        try:
            body = await request.json()
            email = body.get("email") or body.get("username")
            password = body.get("password")
        except Exception:
            pass

    if not email or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both email/username and password are required.",
        )

    # Normalize email to lower case
    email = str(email).strip().lower()

    # Look up user in database
    user = db.query(User).filter(User.email == email).first()

    # Verify password if user exists
    if not user or not verify_password(password, user.password_hash):
        # Record failed login attempt in audit_logs
        record_audit_log(
            db=db,
            action="USER_LOGIN_FAILED",
            user_id=user.id if user else None,
            details={
                "attempted_email": email,
                "reason": "invalid_credentials",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Successful login: record in audit_logs
    role_str = user.role.value if isinstance(user.role, UserRole) else str(user.role)
    record_audit_log(
        db=db,
        action="USER_LOGIN_SUCCESS",
        user_id=user.id,
        details={
            "email": user.email,
            "role": role_str,
        },
    )

    # Generate JWT token
    access_token = create_access_token(
        data={
            "sub": user.email,
            "user_id": user.id,
            "role": role_str,
        }
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        role=role_str,
        user_id=user.id,
    )
