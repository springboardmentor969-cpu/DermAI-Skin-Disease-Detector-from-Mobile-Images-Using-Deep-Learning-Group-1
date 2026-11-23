"""
Authentication router: user registration and JWT token login.
Logs all events (registration, login success, login failure) to the audit_logs table.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User, UserRole
from app.models.refresh_token import RefreshToken
from app.schemas.user import UserCreate, UserResponse, Token, RefreshTokenRequest, LogoutRequest
from app.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    decode_access_token,
    record_audit_log,
    REFRESH_TOKEN_EXPIRE_DAYS,
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

    # Generate JWT access token (30 min) and refresh token (7 days)
    access_token = create_access_token(
        data={
            "sub": user.email,
            "user_id": user.id,
            "role": role_str,
        }
    )
    refresh_token = create_refresh_token(
        data={
            "sub": user.email,
            "user_id": user.id,
            "role": role_str,
        }
    )

    # Persist refresh token in database for revocation tracking
    refresh_expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    db_refresh_token = RefreshToken(
        token=refresh_token,
        user_id=user.id,
        expires_at=refresh_expires_at,
        revoked=False,
    )
    db.add(db_refresh_token)
    db.commit()

    return Token(
        access_token=access_token,
        token_type="bearer",
        role=role_str,
        user_id=user.id,
        refresh_token=refresh_token,
    )


@router.post(
    "/refresh",
    response_model=Token,
    summary="Refresh access token",
    description="Validates a refresh token against the database and returns a new access token.",
)
async def refresh_access_token(
    request: Request,
    body: Optional[RefreshTokenRequest] = None,
    db: Session = Depends(get_db),
):
    token_str: Optional[str] = None
    if body and body.refresh_token:
        token_str = body.refresh_token
    else:
        # Check form data or json body
        try:
            content_type = request.headers.get("content-type", "")
            if "application/x-www-form-urlencoded" in content_type:
                form = await request.form()
                token_str = form.get("refresh_token")
            else:
                raw_json = await request.json()
                token_str = raw_json.get("refresh_token")
        except Exception:
            pass

    if not token_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Refresh token is required.",
        )

    # Decode and validate JWT signature & token_type
    payload = decode_refresh_token(token_str)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Look up refresh token in database
    db_token = (
        db.query(RefreshToken)
        .filter(RefreshToken.token == token_str)
        .first()
    )

    if not db_token or db_token.revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked or is invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check expiration date against current time
    now_utc = datetime.now(timezone.utc)
    # Ensure tz-aware comparison
    token_expires = db_token.expires_at
    if token_expires.tzinfo is None:
        token_expires = token_expires.replace(tzinfo=timezone.utc)

    if token_expires < now_utc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Look up user
    user = db.query(User).filter(User.id == db_token.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with refresh token no longer exists.",
        )

    role_str = user.role.value if isinstance(user.role, UserRole) else str(user.role)

    # Issue new access token
    new_access_token = create_access_token(
        data={
            "sub": user.email,
            "user_id": user.id,
            "role": role_str,
        }
    )

    return Token(
        access_token=new_access_token,
        token_type="bearer",
        role=role_str,
        user_id=user.id,
        refresh_token=token_str,
    )


@router.post(
    "/logout",
    summary="Revoke refresh token and log out user",
    description="Blacklists/revokes the provided refresh token so it cannot be used again.",
)
async def logout(
    request: Request,
    body: Optional[LogoutRequest] = None,
    db: Session = Depends(get_db),
):
    token_str: Optional[str] = None
    if body and body.refresh_token:
        token_str = body.refresh_token
    else:
        try:
            content_type = request.headers.get("content-type", "")
            if "application/x-www-form-urlencoded" in content_type:
                form = await request.form()
                token_str = form.get("refresh_token")
            else:
                raw_json = await request.json()
                token_str = raw_json.get("refresh_token")
        except Exception:
            pass

    user_id = None
    # If refresh token was provided, revoke it in DB
    if token_str:
        db_token = (
            db.query(RefreshToken)
            .filter(RefreshToken.token == token_str)
            .first()
        )
        if db_token:
            db_token.revoked = True
            user_id = db_token.user_id
            db.commit()

    # Also check if request has Authorization bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        access_token = auth_header.split(" ")[1]
        access_payload = decode_access_token(access_token)
        if access_payload:
            user_id = access_payload.get("user_id") or user_id
            # Also revoke all active refresh tokens for this user if no specific token given
            if not token_str and user_id:
                db.query(RefreshToken).filter(
                    RefreshToken.user_id == user_id,
                    RefreshToken.revoked == False,
                ).update({"revoked": True})
                db.commit()

    record_audit_log(
        db=db,
        action="USER_LOGOUT",
        user_id=user_id,
        details={"revoked": True},
    )

    return {"message": "Successfully logged out and session revoked."}

