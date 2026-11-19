"""
Authentication test suite:
- Register works
- Duplicate email fails
- Registration as admin is blocked
- Wrong password fails
- Protected route fails without token
- Protected route succeeds with valid token
- Role-based access control with require_role
- Audit logs are written for register, login, and failed login
"""

from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.models.audit import AuditLog
from app.models.user import UserRole
from app.dependencies import require_role


# Add a dummy protected route to test require_role dependency
@app.get("/test/clinician-only", tags=["Testing"])
def clinician_endpoint(user=Depends(require_role(UserRole.CLINICIAN, UserRole.ADMIN))):
    return {"message": "Welcome clinician", "user": user.email}


def test_register_works(client: TestClient, db_session: Session):
    """Registering a new user succeeds and creates an audit log entry."""
    response = client.post(
        "/auth/register",
        json={
            "email": "sarah.connor@example.com",
            "password": "Password123!",
            "full_name": "Sarah Connor",
            "role": "patient",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "sarah.connor@example.com"
    assert data["full_name"] == "Sarah Connor"
    assert data["role"] == "patient"
    assert "id" in data
    assert "password" not in data

    # Verify audit log was recorded
    audit = db_session.query(AuditLog).filter(AuditLog.action == "USER_REGISTER").first()
    assert audit is not None
    assert audit.details["email"] == "sarah.connor@example.com"


def test_register_admin_forbidden(client: TestClient, db_session: Session):
    """Attempting to register as 'admin' via public endpoint must be rejected with 403."""
    response = client.post(
        "/auth/register",
        json={
            "email": "hacker@example.com",
            "password": "Password123!",
            "full_name": "Bad Actor",
            "role": "admin",
        },
    )
    assert response.status_code == 403
    assert "administrator is not permitted" in response.json()["detail"].lower()

    # Verify blocked audit log
    audit = (
        db_session.query(AuditLog)
        .filter(AuditLog.action == "USER_REGISTER_BLOCKED")
        .first()
    )
    assert audit is not None
    assert audit.details["attempted_role"] == "admin"


def test_duplicate_email_fails(client: TestClient):
    """Registering with an already registered email must return 400 Bad Request."""
    user_payload = {
        "email": "duplicate@example.com",
        "password": "SecurePassword1!",
        "full_name": "Duplicate Tester",
        "role": "clinician",
    }
    # First registration
    r1 = client.post("/auth/register", json=user_payload)
    assert r1.status_code == 201

    # Second registration with same email
    r2 = client.post("/auth/register", json=user_payload)
    assert r2.status_code == 400
    assert "already registered" in r2.json()["detail"].lower()


def test_login_success(client: TestClient, db_session: Session):
    """Successful login returns a JWT access token and logs USER_LOGIN_SUCCESS."""
    # Register
    client.post(
        "/auth/register",
        json={
            "email": "login.test@example.com",
            "password": "MySecretPassword123!",
            "full_name": "Login Tester",
            "role": "clinician",
        },
    )

    # Login with JSON
    login_resp = client.post(
        "/auth/login",
        json={
            "email": "login.test@example.com",
            "password": "MySecretPassword123!",
        },
    )
    assert login_resp.status_code == 200
    token_data = login_resp.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    assert token_data["role"] == "clinician"

    # Verify login audit log
    audit = (
        db_session.query(AuditLog)
        .filter(AuditLog.action == "USER_LOGIN_SUCCESS")
        .first()
    )
    assert audit is not None
    assert audit.details["email"] == "login.test@example.com"


def test_wrong_password_fails(client: TestClient, db_session: Session):
    """Logging in with an invalid password returns 401 and logs USER_LOGIN_FAILED."""
    # Register
    client.post(
        "/auth/register",
        json={
            "email": "authfail@example.com",
            "password": "CorrectPassword1!",
            "full_name": "Fail Tester",
            "role": "patient",
        },
    )

    # Attempt login with wrong password
    response = client.post(
        "/auth/login",
        json={
            "email": "authfail@example.com",
            "password": "WrongPassword!",
        },
    )
    assert response.status_code == 401
    assert "incorrect email or password" in response.json()["detail"].lower()

    # Verify failed audit log
    audit = (
        db_session.query(AuditLog)
        .filter(AuditLog.action == "USER_LOGIN_FAILED")
        .first()
    )
    assert audit is not None
    assert audit.details["attempted_email"] == "authfail@example.com"


def test_protected_route_without_token_fails(client: TestClient):
    """Accessing /users/me without Bearer token returns 401 Unauthorized."""
    response = client.get("/users/me")
    assert response.status_code == 401


def test_protected_route_with_valid_token(client: TestClient):
    """Accessing /users/me with valid Bearer token returns user profile."""
    # Register & Login
    client.post(
        "/auth/register",
        json={
            "email": "me.test@example.com",
            "password": "ValidPassword999!",
            "full_name": "Me Tester",
            "role": "researcher",
        },
    )
    login_resp = client.post(
        "/auth/login",
        json={
            "email": "me.test@example.com",
            "password": "ValidPassword999!",
        },
    )
    token = login_resp.json()["access_token"]

    # Access /users/me
    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    profile = response.json()
    assert profile["email"] == "me.test@example.com"
    assert profile["full_name"] == "Me Tester"
    assert profile["role"] == "researcher"


def test_require_role_dependency(client: TestClient):
    """require_role allows authorized roles and rejects unauthorized roles with 403."""
    # Register a clinician
    client.post(
        "/auth/register",
        json={
            "email": "clinician.allowed@example.com",
            "password": "Password1!",
            "role": "clinician",
        },
    )
    t_clinician = client.post(
        "/auth/login",
        json={"email": "clinician.allowed@example.com", "password": "Password1!"},
    ).json()["access_token"]

    # Register a patient
    client.post(
        "/auth/register",
        json={
            "email": "patient.denied@example.com",
            "password": "Password1!",
            "role": "patient",
        },
    )
    t_patient = client.post(
        "/auth/login",
        json={"email": "patient.denied@example.com", "password": "Password1!"},
    ).json()["access_token"]

    # Clinician accesses /test/clinician-only -> 200 OK
    resp_ok = client.get(
        "/test/clinician-only",
        headers={"Authorization": f"Bearer {t_clinician}"},
    )
    assert resp_ok.status_code == 200

    # Patient accesses /test/clinician-only -> 403 Forbidden
    resp_denied = client.get(
        "/test/clinician-only",
        headers={"Authorization": f"Bearer {t_patient}"},
    )
    assert resp_denied.status_code == 403
    assert "insufficient permissions" in resp_denied.json()["detail"].lower()
