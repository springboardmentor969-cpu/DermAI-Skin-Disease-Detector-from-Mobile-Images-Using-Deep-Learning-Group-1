"""
Core security utilities: password hashing, JWT token creation, and token verification.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import bcrypt
from passlib.context import CryptContext
import jwt

from app.config import settings

# Ensure compatibility between modern bcrypt (>=4.1 / 5.0) and passlib
if not hasattr(bcrypt, "__about__"):
    class _About:
        __version__ = getattr(bcrypt, "__version__", "5.0.0")
    bcrypt.__about__ = _About()

_orig_hashpw = bcrypt.hashpw


def _safe_hashpw(password, salt):
    if len(password) > 72:
        password = password[:72]
    return _orig_hashpw(password, salt)


bcrypt.hashpw = _safe_hashpw

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
REFRESH_TOKEN_EXPIRE_DAYS: int = 7


def hash_password(password: str) -> str:
    """Hash plain password using bcrypt."""
    return pwd_context.hash(password[:72])


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against hashed password."""
    return pwd_context.verify(plain_password[:72], hashed_password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Creates a signed JWT access token.
    Defaults to 30 minutes expiration.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": now,
        "token_type": "access",
    })
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Creates a signed JWT refresh token.
    Defaults to 7 days expiration.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({
        "exp": expire,
        "iat": now,
        "token_type": "refresh",
    })
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and validates an access token.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("token_type") and payload.get("token_type") != "access":
            return None
        return payload
    except jwt.PyJWTError:
        return None


def decode_refresh_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and validates a refresh token.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("token_type") != "refresh":
            return None
        return payload
    except jwt.PyJWTError:
        return None
