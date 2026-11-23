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
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
)


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
