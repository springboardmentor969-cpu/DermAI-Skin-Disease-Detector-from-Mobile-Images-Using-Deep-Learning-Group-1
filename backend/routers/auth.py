from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from passlib.context import CryptContext
from jose import jwt

from app.config import settings
from app.database import users_collection, sessions_collection
from models.user import create_user_document
from schemas.user import (
    UserRegister,
    UserLogin,
    UserResponse,
    TokenResponse
)
from app.dependencies.auth import get_current_user


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


# ============================================================
# PASSWORD HASHING
# ============================================================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str
) -> bool:

    return pwd_context.verify(
        plain_password,
        hashed_password
    )


# ============================================================
# JWT
# ============================================================

def create_access_token(
    user_id: str,
    session_id: str
):

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": user_id,
        "session_id": session_id,
        "exp": expire
    }

    return jwt.encode(
        payload,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )


# ============================================================
# REGISTER
# ============================================================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED
)
def register_user(user: UserRegister):

    email = user.email.lower()

    # Check existing email
    existing_user = users_collection.find_one(
        {"email": email}
    )

    if existing_user:

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email is already registered."
        )

    # Hash password
    password_hash = hash_password(
        user.password
    )

    # Create MongoDB document
    user_document = create_user_document(
        full_name=user.full_name,
        email=email,
        password_hash=password_hash,
        role=user.role
    )

    # Insert user
    result = users_collection.insert_one(
        user_document
    )

    return UserResponse(
        id=str(result.inserted_id),
        full_name=user_document["full_name"],
        email=user_document["email"],
        role=user_document["role"]
    )


# ============================================================
# LOGIN
# ============================================================

@router.post(
    "/login",
    response_model=TokenResponse
)
def login_user(user: UserLogin):

    email = user.email.lower()

    # Find user
    database_user = users_collection.find_one(
        {"email": email}
    )

    if not database_user:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    # Verify password
    password_valid = verify_password(
        user.password,
        database_user["password_hash"]
    )

    if not password_valid:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    # ========================================================
    # CREATE SESSION
    # ========================================================

    session_id = str(uuid4())

    now = datetime.now(timezone.utc)

    users_collection.update_one(
        {"_id": database_user["_id"]},
        {
            "$set": {
                "last_seen": now
            }
        }
    )

    sessions_collection.insert_one({
        "session_id": session_id,
        "user_id": database_user["_id"],
        "created_at": now,
        "last_seen": now,
        "is_active": True
    })

    # ========================================================
    # CREATE JWT
    # ========================================================

    access_token = create_access_token(
        str(database_user["_id"]),
        session_id
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer"
    )


# ============================================================
# LOGOUT
# ============================================================

@router.post("/logout")
def logout_user(
    current_user=Depends(get_current_user)
):

    session_id = current_user.get("_session_id")

    if session_id:

        sessions_collection.update_one(
            {
                "session_id": session_id,
                "user_id": current_user["_id"]
            },
            {
                "$set": {
                    "is_active": False,
                    "logged_out_at": datetime.now(timezone.utc)
                }
            }
        )

    return {
        "message": "Logged out successfully."
    }


# ============================================================
# MY PROFILE
# ============================================================

@router.get(
    "/me",
    response_model=UserResponse
)
def get_my_profile(
    current_user=Depends(get_current_user)
):

    return UserResponse(
        id=str(current_user["_id"]),
        full_name=current_user["full_name"],
        email=current_user["email"],
        role=current_user["role"]
    )