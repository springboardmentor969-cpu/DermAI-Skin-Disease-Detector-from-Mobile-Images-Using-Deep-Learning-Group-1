from datetime import datetime, timezone


def create_user_document(
    full_name: str,
    email: str,
    password_hash: str,
    role: str
):
    now = datetime.now(timezone.utc)

    return {
        "full_name": full_name,
        "email": email.lower(),
        "password_hash": password_hash,
        "role": role,
        "created_at": now,
        "updated_at": now
    }