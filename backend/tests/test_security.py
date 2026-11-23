"""
Tests for DermAI Production Security Hardening:
1. Rate limiting (general endpoints and per-user upload limit)
2. Input validation (magic bytes, MIME types, file size, HTTP 415)
3. JWT access/refresh token lifecycle, revocation, and logout
4. Security response headers (nosniff, DENY, X-XSS-Protection)
5. CORS configuration
"""

import io
from datetime import datetime, timedelta, timezone
from PIL import Image as PILImage
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
)
from app.core.limiter import limiter
from app.models.refresh_token import RefreshToken


def create_sample_image_bytes(fmt="JPEG", size=(100, 100)) -> bytes:
    buf = io.BytesIO()
    img = PILImage.new("RGB", size, color=(150, 100, 70))
    img.save(buf, format=fmt)
    return buf.getvalue()


# ------------------------------------------------------------------------------
# 1. Rate Limiting Tests
# ------------------------------------------------------------------------------

def test_rate_limiting_upload_per_user(client: TestClient):
    """
    POST /images/upload is limited to 10 uploads per minute per user.
    The 11th request receives HTTP 429: 'Too many requests — please wait before uploading again'.
    """
    limiter.reset()

    # Register and login user
    client.post(
        "/auth/register",
        json={
            "email": "rate_user@example.com",
            "password": "Password123!",
            "full_name": "Rate User",
            "role": "patient",
        },
    )
    login_res = client.post(
        "/auth/login",
        json={"email": "rate_user@example.com", "password": "Password123!"},
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    img_bytes = create_sample_image_bytes("JPEG", (100, 100))

    # Send 10 uploads (allowed)
    for i in range(10):
        res = client.post(
            "/images/upload",
            files={"file": (f"test_{i}.jpg", img_bytes, "image/jpeg")},
            headers=headers,
        )
        assert res.status_code in (200, 201), f"Upload {i+1} failed with {res.status_code}"

    # 11th upload should trigger rate limiting (HTTP 429)
    res_11 = client.post(
        "/images/upload",
        files={"file": ("test_11.jpg", img_bytes, "image/jpeg")},
        headers=headers,
    )
    assert res_11.status_code == 429
    assert res_11.json()["detail"] == "Too many requests — please wait before uploading again"

    limiter.reset()


def test_rate_limiting_general_endpoints(client: TestClient):
    """
    General endpoints return HTTP 429 after 100 requests per minute per IP.
    """
    limiter.reset()

    # Fire 100 requests to general endpoint /
    for i in range(100):
        res = client.get("/")
        assert res.status_code == 200

    # 101st request should hit 429
    res_101 = client.get("/")
    assert res_101.status_code == 429
    assert "too many requests" in res_101.json()["detail"].lower()

    limiter.reset()


# ------------------------------------------------------------------------------
# 2. Input Validation Tests (MIME & Magic Bytes)
# ------------------------------------------------------------------------------

def get_test_token(client: TestClient, email: str = "testsec@example.com") -> str:
    """Helper to register and login a test user."""
    client.post(
        "/auth/register",
        json={
            "email": email,
            "password": "Password123!",
            "full_name": "Security Test User",
            "role": "patient",
        },
    )
    login_res = client.post(
        "/auth/login",
        json={"email": email, "password": "Password123!"},
    )
    return login_res.json()["access_token"]


def test_image_upload_supported_formats(client: TestClient):
    """Accepts valid JPEG, PNG, WebP, and BMP images."""
    token = get_test_token(client, "format_user_1@example.com")
    auth_header = {"Authorization": f"Bearer {token}"}

    # 1. JPEG
    jpeg_bytes = create_sample_image_bytes("JPEG")
    r_jpeg = client.post(
        "/images/upload",
        files={"file": ("scan.jpg", jpeg_bytes, "image/jpeg")},
        headers=auth_header,
    )
    assert r_jpeg.status_code == 201

    # 2. PNG
    png_bytes = create_sample_image_bytes("PNG")
    r_png = client.post(
        "/images/upload",
        files={"file": ("scan.png", png_bytes, "image/png")},
        headers=auth_header,
    )
    assert r_png.status_code == 201

    # 3. BMP
    bmp_bytes = create_sample_image_bytes("BMP")
    r_bmp = client.post(
        "/images/upload",
        files={"file": ("scan.bmp", bmp_bytes, "image/bmp")},
        headers=auth_header,
    )
    assert r_bmp.status_code == 201

    # 4. WebP
    webp_bytes = create_sample_image_bytes("WEBP")
    r_webp = client.post(
        "/images/upload",
        files={"file": ("scan.webp", webp_bytes, "image/webp")},
        headers=auth_header,
    )
    assert r_webp.status_code == 201


def test_image_upload_rejects_non_image_mime(client: TestClient):
    """Rejects disallowed MIME type (e.g. application/pdf) with HTTP 415."""
    token = get_test_token(client, "format_user_2@example.com")
    auth_header = {"Authorization": f"Bearer {token}"}

    dummy_pdf = b"%PDF-1.5 fake pdf document content bytes"
    res = client.post(
        "/images/upload",
        files={"file": ("report.pdf", dummy_pdf, "application/pdf")},
        headers=auth_header,
    )
    assert res.status_code == 415
    assert "Only JPEG, PNG, WebP, BMP images accepted" in res.json()["detail"]


def test_image_upload_rejects_fake_extension_magic_bytes(client: TestClient):
    """
    Rejects a non-image file disguised as image/jpeg (wrong magic bytes) with HTTP 415.
    """
    token = get_test_token(client, "format_user_3@example.com")
    auth_header = {"Authorization": f"Bearer {token}"}

    # Plain text masquerading as JPEG
    disguised_text = b"This is just ASCII text without image magic header"
    res = client.post(
        "/images/upload",
        files={"file": ("malicious.jpg", disguised_text, "image/jpeg")},
        headers=auth_header,
    )
    assert res.status_code == 415
    assert "Only JPEG, PNG, WebP, BMP images accepted" in res.json()["detail"]


def test_image_upload_rejects_over_20mb(client: TestClient):
    """Rejects file exceeding 20MB with HTTP 413."""
    token = get_test_token(client, "format_user_4@example.com")
    auth_header = {"Authorization": f"Bearer {token}"}

    # Header starts with JPEG magic bytes, but size > 20MB
    huge_payload = b"\xff\xd8\xff\xe0" + b"X" * (20 * 1024 * 1024 + 100)
    res = client.post(
        "/images/upload",
        files={"file": ("huge.jpg", huge_payload, "image/jpeg")},
        headers=auth_header,
    )
    assert res.status_code == 413


# ------------------------------------------------------------------------------
# 3. JWT Access & Refresh Token Tests
# ------------------------------------------------------------------------------

def test_jwt_expiry_constants():
    """Validates configured token expiration times."""
    assert ACCESS_TOKEN_EXPIRE_MINUTES == 30
    assert REFRESH_TOKEN_EXPIRE_DAYS == 7


def test_jwt_login_returns_refresh_token(client: TestClient, db_session: Session):
    """POST /auth/login returns both access_token and refresh_token, saved in DB."""
    client.post(
        "/auth/register",
        json={
            "email": "jwttest1@example.com",
            "password": "Password123!",
            "full_name": "JWT Test 1",
            "role": "patient",
        },
    )

    login_res = client.post(
        "/auth/login",
        json={"email": "jwttest1@example.com", "password": "Password123!"},
    )
    assert login_res.status_code == 200
    data = login_res.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["refresh_token"] is not None

    # Verify refresh token in database
    db_token = db_session.query(RefreshToken).filter(RefreshToken.token == data["refresh_token"]).first()
    assert db_token is not None
    assert db_token.revoked is False
    expires_dt = db_token.expires_at
    if expires_dt.tzinfo is None:
        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
    assert expires_dt > datetime.now(timezone.utc)


def test_jwt_refresh_and_logout_flow(client: TestClient, db_session: Session):
    """
    POST /auth/refresh exchanges refresh token for new access token.
    POST /auth/logout revokes refresh token, preventing further refreshes.
    """
    client.post(
        "/auth/register",
        json={
            "email": "jwttest2@example.com",
            "password": "Password123!",
            "full_name": "JWT Test 2",
            "role": "patient",
        },
    )
    login_res = client.post(
        "/auth/login",
        json={"email": "jwttest2@example.com", "password": "Password123!"},
    )
    initial_tokens = login_res.json()
    ref_token = initial_tokens["refresh_token"]

    # 1. Refresh token successfully
    refresh_res = client.post(
        "/auth/refresh",
        json={"refresh_token": ref_token},
    )
    assert refresh_res.status_code == 200
    refreshed_data = refresh_res.json()
    assert "access_token" in refreshed_data
    assert len(refreshed_data["access_token"]) > 20

    # 2. Logout revoking the refresh token
    logout_res = client.post(
        "/auth/logout",
        json={"refresh_token": ref_token},
    )
    assert logout_res.status_code == 200

    # 3. Attempting refresh with revoked token should fail with HTTP 401
    fail_res = client.post(
        "/auth/refresh",
        json={"refresh_token": ref_token},
    )
    assert fail_res.status_code == 401
    assert "revoked" in fail_res.json()["detail"].lower()


# ------------------------------------------------------------------------------
# 4. Security Headers & CORS Tests
# ------------------------------------------------------------------------------

def test_security_headers_present_on_all_responses(client: TestClient):
    """Every HTTP response includes nosniff, DENY, and XSS protection headers."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert res.headers.get("X-XSS-Protection") == "1; mode=block"

    # Also check a 404 response
    res_404 = client.get("/nonexistent-endpoint-abc")
    assert res_404.headers.get("X-Content-Type-Options") == "nosniff"
    assert res_404.headers.get("X-Frame-Options") == "DENY"
    assert res_404.headers.get("X-XSS-Protection") == "1; mode=block"


def test_cors_development_origins(client: TestClient):
    """CORS allows origin http://localhost:5173 and http://localhost:3000."""
    res = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"
