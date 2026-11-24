from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from bson import ObjectId

from app.config import settings
from app.database import users_collection, sessions_collection


security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    token = credentials.credentials

    # ========================================================
    # DECODE JWT
    # ========================================================

    try:

        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )

        user_id = payload.get("sub")
        session_id = payload.get("session_id")

        if not user_id or not session_id:

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token."
            )

    except JWTError:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token."
        )

    # ========================================================
    # CONVERT USER ID
    # ========================================================

    try:

        object_id = ObjectId(user_id)

    except Exception:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID."
        )

    # ========================================================
    # FIND USER
    # ========================================================

    user = users_collection.find_one(
        {"_id": object_id}
    )

    if not user:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found."
        )

    # ========================================================
    # CHECK SESSION
    # ========================================================

    session = sessions_collection.find_one({
        "session_id": session_id,
        "user_id": object_id,
        "is_active": True
    })

    if not session:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session is no longer active."
        )

    # ========================================================
    # UPDATE LAST SEEN
    # ========================================================

    now = datetime.now(timezone.utc)

    users_collection.update_one(
        {"_id": object_id},
        {
            "$set": {
                "last_seen": now
            }
        }
    )

    sessions_collection.update_one(
        {
            "session_id": session_id,
            "user_id": object_id
        },
        {
            "$set": {
                "last_seen": now
            }
        }
    )

    # ========================================================
    # ATTACH SESSION ID
    # ========================================================

    user["_session_id"] = session_id

    return user