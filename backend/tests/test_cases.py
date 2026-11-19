"""
Cases and image upload test suite:
- Create case with patient demographic info
- Role-based case visibility (patient only sees their own, clinician sees all)
- Valid image upload (JPG / PNG, resolution >= 224x224)
- Wrong file type rejection (corrupt files, non-images, disallow formats like GIF)
- Too-large file rejection (> 10 MB)
- Too-low resolution rejection (< 224x224)
- Audit log verification for IMAGE_UPLOADED
"""

import io
import pytest
from PIL import Image as PILImage
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.audit import AuditLog


def create_test_image(
    width: int = 300,
    height: int = 300,
    img_format: str = "JPEG",
    color: str = "red",
) -> bytes:
    """Generate in-memory image bytes for testing."""
    buf = io.BytesIO()
    img = PILImage.new("RGB", (width, height), color=color)
    img.save(buf, format=img_format)
    return buf.getvalue()


def get_auth_token(client: TestClient, email: str, role: str) -> str:
    """Helper to register and login a user and return the JWT access token."""
    client.post(
        "/auth/register",
        json={
            "email": email,
            "password": "Password123!",
            "full_name": f"User {email}",
            "role": role,
        },
    )
    login_res = client.post(
        "/auth/login",
        json={"email": email, "password": "Password123!"},
    )
    return login_res.json()["access_token"]


def test_create_and_get_case(client: TestClient):
    """Create a case and retrieve it by ID."""
    token = get_auth_token(client, "patient1@example.com", "patient")

    case_payload = {
        "body_location": "Left forearm",
        "status": "submitted",
        "patient_age": 42,
        "patient_gender": "female",
        "clinical_notes": "Mild erythema observed",
    }
    create_res = client.post(
        "/cases",
        json=case_payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert create_res.status_code == 201
    case_data = create_res.json()
    assert case_data["body_location"] == "Left forearm"
    assert case_data["patient"]["age"] == 42
    assert case_data["patient"]["gender"] == "female"
    case_id = case_data["id"]

    # Get case by ID
    get_res = client.get(
        f"/cases/{case_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert get_res.status_code == 200
    assert get_res.json()["id"] == case_id


def test_case_visibility_patient_vs_clinician(client: TestClient):
    """Patients see only their own cases; clinicians see all cases."""
    token_pat1 = get_auth_token(client, "pat1@example.com", "patient")
    token_pat2 = get_auth_token(client, "pat2@example.com", "patient")
    token_doc = get_auth_token(client, "doc1@example.com", "clinician")

    # Patient 1 creates a case
    res1 = client.post(
        "/cases",
        json={"body_location": "Back", "patient_age": 30},
        headers={"Authorization": f"Bearer {token_pat1}"},
    )
    case1_id = res1.json()["id"]

    # Patient 2 creates a case
    res2 = client.post(
        "/cases",
        json={"body_location": "Cheek", "patient_age": 55},
        headers={"Authorization": f"Bearer {token_pat2}"},
    )
    case2_id = res2.json()["id"]

    # Patient 1 lists cases -> sees only case 1
    pat1_list = client.get("/cases", headers={"Authorization": f"Bearer {token_pat1}"}).json()
    pat1_ids = [c["id"] for c in pat1_list]
    assert case1_id in pat1_ids
    assert case2_id not in pat1_ids

    # Patient 1 tries to view case 2 -> 403 Forbidden
    denied = client.get(f"/cases/{case2_id}", headers={"Authorization": f"Bearer {token_pat1}"})
    assert denied.status_code == 403

    # Clinician lists cases -> sees both case 1 and case 2
    doc_list = client.get("/cases", headers={"Authorization": f"Bearer {token_doc}"}).json()
    doc_ids = [c["id"] for c in doc_list]
    assert case1_id in doc_ids
    assert case2_id in doc_ids


def test_valid_image_upload(client: TestClient, db_session: Session):
    """Uploading a valid JPEG image (>= 224x224, <= 10MB) succeeds and logs IMAGE_UPLOADED."""
    token = get_auth_token(client, "uploader@example.com", "patient")

    case_res = client.post(
        "/cases",
        json={"body_location": "Right shoulder", "patient_age": 45},
        headers={"Authorization": f"Bearer {token}"},
    )
    case_id = case_res.json()["id"]

    # Generate a valid 300x300 JPEG image
    image_bytes = create_test_image(width=300, height=300, img_format="JPEG")

    files = {
        "file": ("lesion_scan.jpg", image_bytes, "image/jpeg"),
    }
    data = {"source": "camera"}

    upload_res = client.post(
        f"/cases/{case_id}/images",
        files=files,
        data=data,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert upload_res.status_code == 201
    img_data = upload_res.json()
    assert img_data["case_id"] == case_id
    assert "cases/" in img_data["storage_key"]
    assert img_data["source"] == "camera"
    assert img_data["quality_details"]["width"] == 300
    assert img_data["quality_details"]["height"] == 300

    # Verify audit log was recorded
    audit = (
        db_session.query(AuditLog)
        .filter(AuditLog.action == "IMAGE_UPLOADED")
        .first()
    )
    assert audit is not None
    assert audit.details["case_id"] == case_id
    assert audit.details["dimensions"] == "300x300"
    assert "quality_label" in img_data


def test_poor_quality_image_saved_with_warning(client: TestClient, db_session: Session):
    """A poor quality (blurry) image is still saved to the DB, but returns a warning to retake."""
    import cv2
    import numpy as np

    token = get_auth_token(client, "poorimage@example.com", "patient")

    case_res = client.post(
        "/cases",
        json={"body_location": "Forearm"},
        headers={"Authorization": f"Bearer {token}"},
    )
    case_id = case_res.json()["id"]

    # Generate a heavily blurred 300x300 image
    sharp_bytes = create_test_image(width=300, height=300, img_format="JPEG")
    # Blur heavily
    sharp_img = PILImage.open(io.BytesIO(sharp_bytes))
    blurred_np = cv2.GaussianBlur(np.array(sharp_img), (55, 55), 0)
    buf = io.BytesIO()
    PILImage.fromarray(blurred_np).save(buf, format="JPEG")
    blurred_bytes = buf.getvalue()

    upload_res = client.post(
        f"/cases/{case_id}/images",
        files={"file": ("blurry_lesion.jpg", blurred_bytes, "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"},
    )
    # Poor images MUST still be saved with 201 Created
    assert upload_res.status_code == 201
    img_data = upload_res.json()
    assert img_data["quality_label"] == "poor"
    assert img_data["warning"] is not None
    assert "retake" in img_data["warning"].lower()
    assert img_data["storage_key"] is not None

    # List images for case
    list_images_res = client.get(
        f"/cases/{case_id}/images",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_images_res.status_code == 200
    assert len(list_images_res.json()) >= 1


def test_wrong_file_type_fails(client: TestClient):
    """Uploading a non-image file or invalid format (e.g. GIF or text disguised as jpg) fails."""
    token = get_auth_token(client, "badtype@example.com", "patient")

    case_res = client.post(
        "/cases",
        json={"body_location": "Neck"},
        headers={"Authorization": f"Bearer {token}"},
    )
    case_id = case_res.json()["id"]

    # 1. Plain text masquerading as JPG
    fake_img = b"Not a real JPEG image binary content at all"
    r1 = client.post(
        f"/cases/{case_id}/images",
        files={"file": ("fake.jpg", fake_img, "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r1.status_code == 400
    assert "invalid image file" in r1.json()["detail"].lower()

    # 2. Valid image but disallowed format (GIF)
    gif_bytes = create_test_image(width=250, height=250, img_format="GIF")
    r2 = client.post(
        f"/cases/{case_id}/images",
        files={"file": ("animation.gif", gif_bytes, "image/gif")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r2.status_code == 400
    assert "unsupported image format" in r2.json()["detail"].lower()


def test_too_large_file_fails(client: TestClient):
    """Uploading a file exceeding 10 MB is rejected with 400 Bad Request."""
    token = get_auth_token(client, "toolarge@example.com", "patient")

    case_res = client.post(
        "/cases",
        json={"body_location": "Scalp"},
        headers={"Authorization": f"Bearer {token}"},
    )
    case_id = case_res.json()["id"]

    # Create dummy payload larger than 10 MB (10 MB + 1 KB)
    oversized_data = b"X" * (10 * 1024 * 1024 + 1024)

    response = client.post(
        f"/cases/{case_id}/images",
        files={"file": ("huge_lesion.jpg", oversized_data, "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400
    assert "exceeds maximum allowed size" in response.json()["detail"].lower()


def test_too_low_resolution_fails(client: TestClient):
    """Uploading an image below 224x224 minimum resolution is rejected."""
    token = get_auth_token(client, "lowres@example.com", "patient")

    case_res = client.post(
        "/cases",
        json={"body_location": "Wrist"},
        headers={"Authorization": f"Bearer {token}"},
    )
    case_id = case_res.json()["id"]

    # Image with 150x150 resolution (below 224x224 requirement)
    low_res_bytes = create_test_image(width=150, height=150, img_format="PNG")

    response = client.post(
        f"/cases/{case_id}/images",
        files={"file": ("tiny.png", low_res_bytes, "image/png")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400
    assert "resolution" in response.json()["detail"].lower()
    assert "too low" in response.json()["detail"].lower()


def test_get_image_content(client: TestClient):
    """Retrieving raw binary image content from MinIO via streaming endpoint."""
    token = get_auth_token(client, "streamer@example.com", "patient")

    case_res = client.post(
        "/cases",
        json={"body_location": "Cheek"},
        headers={"Authorization": f"Bearer {token}"},
    )
    case_id = case_res.json()["id"]

    valid_bytes = create_test_image(width=300, height=300, img_format="JPEG")
    upload_res = client.post(
        f"/cases/{case_id}/images",
        files={"file": ("cheek.jpg", valid_bytes, "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"},
    )
    image_id = upload_res.json()["id"]

    # Download image content
    content_res = client.get(
        f"/cases/{case_id}/images/{image_id}/content",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert content_res.status_code == 200
    assert content_res.headers["content-type"] == "image/jpeg"
    assert len(content_res.content) > 0

