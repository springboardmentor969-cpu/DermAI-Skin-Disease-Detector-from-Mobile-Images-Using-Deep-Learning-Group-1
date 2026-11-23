"""
Rate limiter configuration using slowapi for API abuse prevention.
"""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings


def get_user_or_ip(request: Request) -> str:
    """
    Returns user ID from JWT bearer token for per-user rate limiting,
    falling back to remote IP address if unauthenticated.
    """
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        try:
            from app.core.security import decode_access_token
            payload = decode_access_token(token)
            if payload:
                user_id = payload.get("user_id") or payload.get("sub")
                if user_id:
                    return f"user:{user_id}"
        except Exception:
            pass
        if token:
            return f"token:{token[-16:]}"

    return get_remote_address(request)


# Limiter instance with default 100 requests per minute per IP for general endpoints
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute"],
)
