"""
Authentication service providing password hashing, JWT token generation & verification,
and audit trail logging.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import bcrypt
from sqlalchemy.orm import Session

# Ensure compatibility between modern bcrypt (>=4.1 / 5.0) and passlib
if not hasattr(bcrypt, "__about__"):
    class _About:
        __version__ = getattr(bcrypt, "__version__", "5.0.0")
    bcrypt.__about__ = _About()

# Bcrypt enforces a 72-byte max password length. Wrap hashpw to safely truncate
# rather than throwing an unhandled ValueError during passlib's internal wrap check.
_orig_hashpw = bcrypt.hashpw


def _safe_hashpw(password, salt):
    if len(password) > 72:
        password = password[:72]
    return _orig_hashpw(password, salt)


bcrypt.hashpw = _safe_hashpw

from passlib.context import CryptContext
import jwt

from app.config import settings
from app.models.audit import AuditLog

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash plain password using bcrypt."""
    # Truncate at 72 bytes to satisfy bcrypt specification
    return pwd_context.hash(password[:72])


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against hashed password."""
    return pwd_context.verify(plain_password[:72], hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a signed JWT access token with expiration time and payload claims.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": now,
    })
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT access token against the configured secret key.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None


def record_audit_log(
    db: Session,
    action: str,
    user_id: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
) -> AuditLog:
    """
    Persist an audit entry into the audit_logs table.
    """
    log_entry = AuditLog(
        action=action,
        user_id=user_id,
        details=details or {},
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return log_entry
