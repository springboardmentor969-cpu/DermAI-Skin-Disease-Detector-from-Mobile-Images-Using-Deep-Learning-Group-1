"""
Unit and integration tests for Patient Prediction History router.

Tests:
1. GET /patients/{patient_id}/history requires JWT authentication (401).
2. Patients can only view their own history (403 when trying to view another patient's).
3. Clinicians/admins can view any patient's history (200).
4. Returns correct fields: case_id, thumbnail_url, predicted_class, confidence, risk_level, created_at.
5. Pagination (?page=1&limit=2) returns correct slice and pagination headers.
"""

import io
import pytest
from datetime import datetime, timezone
from PIL import Image as PILImage
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.case import Case
from app.models.image import Image
from app.models.patient import Patient
from app.models.prediction import Prediction
from app.services.storage_service import storage_service


def get_auth_token(client: TestClient, email: str, role: str) -> str:
    """Register and login a user to acquire a JWT token."""
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


def test_get_history_unauthenticated(client: TestClient):
    """Unauthenticated request is rejected with 401."""
    res = client.get("/patients/1/history")
    assert res.status_code == 401


def test_get_history_patient_authorization(client: TestClient, db_session: Session):
    """
    Patients can only view their own history.
    Patient 2 attempting to view Patient 1's history receives 403.
    """
    # Create Patient 1 owned by User 1
    p1 = Patient(age=40, gender="female", notes="Patient One")
    db_session.add(p1)
    db_session.commit()

    case1 = Case(patient_id=p1.id, created_by=1, body_location="Arm", status="submitted")
    db_session.add(case1)
    db_session.commit()

    img1 = Image(case_id=case1.id, storage_key="cases/1/scan1.jpg", quality_label="good")
    db_session.add(img1)
    db_session.commit()

    pred1 = Prediction(
        image_id=img1.id,
        model_version="resnet50_v1.0",
        predicted_class="mel",
        confidence=0.88,
        risk_level="HIGH",
    )
    db_session.add(pred1)
    db_session.commit()

    # Create tokens
    pat1_token = get_auth_token(client, "patient_one@example.com", "patient")
    pat2_token = get_auth_token(client, "patient_two@example.com", "patient")

    # Patient 2 tries to access Patient 1's history -> 403 Forbidden
    forbidden_res = client.get(
        f"/patients/{p1.id}/history",
        headers={"Authorization": f"Bearer {pat2_token}"},
    )
    assert forbidden_res.status_code == 403
    assert "Access forbidden" in forbidden_res.json()["detail"]

    # Patient 1 views their own history with 'me' -> 200 OK
    me_res = client.get(
        "/patients/me/history",
        headers={"Authorization": f"Bearer {pat1_token}"},
    )
    assert me_res.status_code == 200
    assert isinstance(me_res.json(), list)


def test_get_history_clinician_access_and_pagination(client: TestClient, db_session: Session, monkeypatch):
    """
    Clinician can view any patient's history.
    Verifies response structure, thumbnail URL, risk level, and pagination.
    """
    clinician_token = get_auth_token(client, "history_clinician@example.com", "clinician")

    # Mock MinIO presigned URL
    monkeypatch.setattr(
        storage_service,
        "get_presigned_url",
        lambda key: f"http://localhost:9000/dermai-images/{key}?signature=abc",
    )

    # Create patient with 3 predictions
    patient = Patient(age=62, gender="male", notes="Longitudinal monitoring")
    db_session.add(patient)
    db_session.commit()

    case = Case(patient_id=patient.id, created_by=1, body_location="Back", status="submitted")
    db_session.add(case)
    db_session.commit()

    for i, (cls_code, conf, risk) in enumerate([
        ("mel", 0.92, "HIGH"),
        ("bcc", 0.65, "HIGH"),
        ("nv", 0.84, "LOW"),
    ]):
        img = Image(case_id=case.id, storage_key=f"cases/1/img_{i}.jpg", quality_label="good")
        db_session.add(img)
        db_session.commit()

        pred = Prediction(
            image_id=img.id,
            model_version="resnet50_v1.0",
            predicted_class=cls_code,
            confidence=conf,
            risk_level=risk,
        )
        db_session.add(pred)
        db_session.commit()

    # Clinician requests page 1 with limit 2
    res_p1 = client.get(
        f"/patients/{patient.id}/history?page=1&limit=2",
        headers={"Authorization": f"Bearer {clinician_token}"},
    )
    assert res_p1.status_code == 200
    data_p1 = res_p1.json()
    assert len(data_p1) == 2
    assert res_p1.headers["x-total-count"] == "3"
    assert res_p1.headers["x-page"] == "1"
    assert res_p1.headers["x-limit"] == "2"

    first_item = data_p1[0]
    assert first_item["case_id"] == case.id
    assert "thumbnail_url" in first_item
    assert "signature=abc" in first_item["thumbnail_url"]
    assert first_item["predicted_class"] in ("mel", "bcc", "nv")
    assert 0.0 <= first_item["confidence"] <= 1.0
    assert first_item["risk_level"] in ("HIGH", "MEDIUM", "LOW")
    assert "created_at" in first_item

    # Clinician requests page 2 with limit 2
    res_p2 = client.get(
        f"/patients/{patient.id}/history?page=2&limit=2",
        headers={"Authorization": f"Bearer {clinician_token}"},
    )
    assert res_p2.status_code == 200
    data_p2 = res_p2.json()
    assert len(data_p2) == 1

