"""
Full API Integration Tests for DermAI.

Covers:
- POST /auth/register — register patient and clinician accounts
- POST /auth/login — login and get JWT token
- POST /images/upload — upload a test image (uses image from ml/data/)
- GET  /images/{image_id} — fetch image metadata
- GET  /images/{image_id}/gradcam?type=heatmap — fetch Grad-CAM heatmap
- GET  /images/{image_id}/gradcam?type=overlay — fetch Grad-CAM overlay
- GET  /patients/{patient_id}/history — fetch prediction history
- GET  /dashboard/stats — fetch clinician dashboard statistics
- GET  /dashboard/class-distribution — class distribution chart data
- GET  /dashboard/cases-over-time — time series data
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_test_image_bytes() -> bytes:
    """Finds a test JPEG image from ml/data or generates a fallback valid 250x250 image."""
    candidates = [
        PROJECT_ROOT / "ml" / "data" / "isic2018_task1" / "images" / "ISIC_0024386.jpg",
        PROJECT_ROOT / "ml" / "preprocessing" / "sample_images" / "01_sharp_good.jpg",
    ]
    for c in candidates:
        if c.is_file():
            return c.read_bytes()

    import io
    from PIL import Image as PILImage
    buf = io.BytesIO()
    img = PILImage.new("RGB", (250, 250), color=(180, 100, 80))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_auth_register(client: TestClient):
    """POST /auth/register registers both patient and clinician accounts with HTTP 201."""
    # 1. Register Patient
    patient_res = client.post(
        "/auth/register",
        json={
            "email": "test_patient_reg@example.com",
            "password": "Password123!",
            "full_name": "Test Patient",
            "role": "patient",
        },
    )
    assert patient_res.status_code == 201
    p_data = patient_res.json()
    assert p_data["email"] == "test_patient_reg@example.com"
    assert p_data["role"] == "patient"
    assert "id" in p_data

    # 2. Register Clinician
    clinician_res = client.post(
        "/auth/register",
        json={
            "email": "test_clinician_reg@example.com",
            "password": "Password123!",
            "full_name": "Dr. Test Clinician",
            "role": "clinician",
        },
    )
    assert clinician_res.status_code == 201
    c_data = clinician_res.json()
    assert c_data["email"] == "test_clinician_reg@example.com"
    assert c_data["role"] == "clinician"


def test_auth_login(client: TestClient):
    """POST /auth/login returns JWT bearer access token for valid credentials."""
    # First register user
    client.post(
        "/auth/register",
        json={
            "email": "login_user@example.com",
            "password": "SecurePassword123!",
            "full_name": "Login User",
            "role": "patient",
        },
    )

    # Login
    login_res = client.post(
        "/auth/login",
        json={"email": "login_user@example.com", "password": "SecurePassword123!"},
    )
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    assert token_data["token_type"].lower() == "bearer"
    assert len(token_data["access_token"]) > 20


def test_images_upload_and_get_metadata(patient_client: TestClient):
    """POST /images/upload uploads a lesion image and GET /images/{id} fetches its metadata."""
    img_bytes = get_test_image_bytes()
    files = {"file": ("lesion_sample.jpg", img_bytes, "image/jpeg")}
    data = {"source": "camera"}

    # 1. Upload via POST /images/upload
    upload_res = patient_client.post("/images/upload", files=files, data=data)
    assert upload_res.status_code == 201
    img_data = upload_res.json()
    image_id = img_data["id"]
    assert image_id > 0
    assert img_data["case_id"] > 0
    assert "storage_key" in img_data
    assert img_data["quality_label"] in ("good", "acceptable", "poor")

    # 2. Fetch metadata via GET /images/{image_id}
    meta_res = patient_client.get(f"/images/{image_id}")
    assert meta_res.status_code == 200
    meta_data = meta_res.json()
    assert meta_data["id"] == image_id
    assert meta_data["case_id"] == img_data["case_id"]
    assert meta_data["storage_key"] == img_data["storage_key"]


def test_images_gradcam_endpoints(patient_client: TestClient):
    """GET /images/{id}/gradcam?type=heatmap and ?type=overlay return PNG image responses."""
    img_bytes = get_test_image_bytes()
    upload_res = patient_client.post(
        "/images/upload",
        files={"file": ("gradcam_test.jpg", img_bytes, "image/jpeg")},
    )
    assert upload_res.status_code == 201
    image_id = upload_res.json()["id"]

    # Analyze image to trigger Grad-CAM generation and storage
    analyze_res = patient_client.post(f"/images/{image_id}/analyze")
    assert analyze_res.status_code == 200

    # 1. GET /images/{image_id}/gradcam?type=heatmap
    heatmap_res = patient_client.get(f"/images/{image_id}/gradcam?type=heatmap")
    assert heatmap_res.status_code == 200
    assert "image/png" in heatmap_res.headers.get("content-type", "")
    assert len(heatmap_res.content) > 0

    # 2. GET /images/{image_id}/gradcam?type=overlay
    overlay_res = patient_client.get(f"/images/{image_id}/gradcam?type=overlay")
    assert overlay_res.status_code == 200
    assert "image/png" in overlay_res.headers.get("content-type", "")
    assert len(overlay_res.content) > 0


def test_patient_history_endpoint(patient_client: TestClient):
    """GET /patients/{patient_id}/history returns paginated analysis timeline."""
    user_id = patient_client.auth_user.id

    # Upload and analyze an image for this patient
    img_bytes = get_test_image_bytes()
    upload_res = patient_client.post(
        "/images/upload",
        files={"file": ("history_test.jpg", img_bytes, "image/jpeg")},
    )
    image_id = upload_res.json()["id"]
    patient_client.post(f"/images/{image_id}/analyze")

    # Fetch patient history
    history_res = patient_client.get(f"/patients/{user_id}/history")
    assert history_res.status_code == 200
    history_items = history_res.json()
    assert isinstance(history_items, list)
    assert len(history_items) >= 1

    entry = history_items[0]
    assert "case_id" in entry
    assert "predicted_class" in entry
    assert "confidence" in entry
    assert "risk_level" in entry
    assert entry["risk_level"] in ("HIGH", "MEDIUM", "LOW")


def test_dashboard_stats(clinician_client: TestClient, patient_client: TestClient):
    """GET /dashboard/stats returns high-level metrics for clinician, rejects patient."""
    # Clinician access succeeds
    res = clinician_client.get("/dashboard/stats")
    assert res.status_code == 200
    stats = res.json()
    for field in [
        "total_cases",
        "cases_this_week",
        "cases_this_month",
        "avg_confidence",
        "high_risk_count",
        "pending_review_count",
    ]:
        assert field in stats
        assert isinstance(stats[field], (int, float))

    # Patient access is forbidden
    forbidden_res = patient_client.get("/dashboard/stats")
    assert forbidden_res.status_code == 403


def test_dashboard_class_distribution(clinician_client: TestClient):
    """GET /dashboard/class-distribution returns 7-class breakdown with counts and percentages."""
    res = clinician_client.get("/dashboard/class-distribution")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) == 7
    class_codes = {item["class_code"] for item in data}
    expected_codes = {"mel", "nv", "bkl", "bcc", "akiec", "vasc", "df"}
    assert class_codes == expected_codes
    for item in data:
        assert "class_name" in item
        assert "count" in item
        assert "percentage" in item
        assert 0.0 <= item["percentage"] <= 100.0


def test_dashboard_cases_over_time(clinician_client: TestClient):
    """GET /dashboard/cases-over-time returns 30-day chronological activity series."""
    res = clinician_client.get("/dashboard/cases-over-time")
    assert res.status_code == 200
    series = res.json()
    assert isinstance(series, list)
    assert len(series) == 30
    for day_entry in series:
        assert "date" in day_entry
        assert "count" in day_entry
        assert isinstance(day_entry["count"], int)
