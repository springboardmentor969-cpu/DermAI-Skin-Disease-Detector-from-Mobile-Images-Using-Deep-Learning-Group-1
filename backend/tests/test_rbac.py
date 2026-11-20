from __future__ import annotations

import io
import uuid
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def test_patient_and_doctor_registration_and_login():
    uid = uuid.uuid4().hex[:8]
    patient_email = f"patient_{uid}@example.com"
    doctor_email = f"doctor_{uid}@example.com"
    password = "SecurePassword123!"

    # 1. Register Patient
    p_reg = client.post(
        "/api/v1/auth/register",
        json={
            "email": patient_email,
            "password": password,
            "full_name": f"Test Patient {uid}",
            "role": "patient",
        },
    )
    assert p_reg.status_code == 201, p_reg.text
    p_reg_data = p_reg.json()
    assert p_reg_data["role"] == "patient"
    assert p_reg_data["email"] == patient_email

    # 2. Register Doctor
    d_reg = client.post(
        "/api/v1/auth/register",
        json={
            "email": doctor_email,
            "password": password,
            "full_name": f"Dr. Specialist {uid}",
            "role": "doctor",
        },
    )
    assert d_reg.status_code == 201, d_reg.text
    d_reg_data = d_reg.json()
    assert d_reg_data["role"] == "doctor"
    assert d_reg_data["email"] == doctor_email

    # 3. Login Patient
    p_login = client.post(
        "/api/v1/auth/login",
        json={"email": patient_email, "password": password},
    )
    assert p_login.status_code == 200
    p_login_data = p_login.json()
    assert p_login_data["role"] == "patient"
    assert "access_token" in p_login_data

    # 4. Login Doctor
    d_login = client.post(
        "/api/v1/auth/login",
        json={"email": doctor_email, "password": password},
    )
    assert d_login.status_code == 200
    d_login_data = d_login.json()
    assert d_login_data["role"] == "doctor"
    assert "access_token" in d_login_data


def test_rbac_case_access_and_clinical_review():
    uid = uuid.uuid4().hex[:8]
    p1_email = f"p1_{uid}@example.com"
    p2_email = f"p2_{uid}@example.com"
    doc_email = f"doc_{uid}@example.com"
    password = "SecurePassword123!"

    # Register p1, p2, doctor
    r1 = client.post("/api/v1/auth/register", json={"email": p1_email, "password": password, "full_name": f"Patient 1 {uid}", "role": "patient"})
    assert r1.status_code == 201, r1.text
    r2 = client.post("/api/v1/auth/register", json={"email": p2_email, "password": password, "full_name": f"Patient 2 {uid}", "role": "patient"})
    assert r2.status_code == 201, r2.text
    rd = client.post("/api/v1/auth/register", json={"email": doc_email, "password": password, "full_name": f"Dr. Dermatologist {uid}", "role": "doctor"})
    assert rd.status_code == 201, rd.text

    p1_token = client.post("/api/v1/auth/login", json={"email": p1_email, "password": password}).json()["access_token"]
    p2_token = client.post("/api/v1/auth/login", json={"email": p2_email, "password": password}).json()["access_token"]
    doc_token = client.post("/api/v1/auth/login", json={"email": doc_email, "password": password}).json()["access_token"]

    p1_headers = {"Authorization": f"Bearer {p1_token}"}
    p2_headers = {"Authorization": f"Bearer {p2_token}"}
    doc_headers = {"Authorization": f"Bearer {doc_token}"}

    # 1. Patient 1 creates a case
    create_resp = client.post(
        "/api/v1/cases",
        headers=p1_headers,
        json={"title": "Suspicious Pigmented Lesion", "description": "Irregular border on left shoulder"},
    )
    assert create_resp.status_code == 201
    case_id = create_resp.json()["id"]

    # 2. Patient 1 can access their own case
    p1_get = client.get(f"/api/v1/cases/{case_id}", headers=p1_headers)
    assert p1_get.status_code == 200
    assert p1_get.json()["id"] == case_id
    assert p1_get.json()["review_status"] == "pending_review"

    # 3. Patient 2 cannot access Patient 1's case (access control check)
    p2_get = client.get(f"/api/v1/cases/{case_id}", headers=p2_headers)
    assert p2_get.status_code == 404, "Patient 2 should not be able to access Patient 1's case"

    # 4. Patient 1 cannot perform a clinical review (only Doctor role is permitted)
    p1_review = client.post(
        f"/api/v1/cases/{case_id}/review",
        headers=p1_headers,
        json={"review_status": "reviewed", "doctor_notes": "Self review not allowed"},
    )
    assert p1_review.status_code == 403, "Patient must receive 403 Forbidden when attempting clinical review"

    # 5. Doctor can access Patient 1's case for clinical evaluation
    doc_get = client.get(f"/api/v1/cases/{case_id}", headers=doc_headers)
    assert doc_get.status_code == 200
    assert doc_get.json()["id"] == case_id

    # 6. Doctor lists cases via triage endpoint
    doc_list = client.get("/api/v1/cases?doctor_view=true", headers=doc_headers)
    assert doc_list.status_code == 200
    triage_ids = [c["id"] for c in doc_list.json()]
    assert case_id in triage_ids

    # 7. Doctor records a clinical review
    review_resp = client.post(
        f"/api/v1/cases/{case_id}/review",
        headers=doc_headers,
        json={
            "review_status": "biopsy_recommended",
            "doctor_notes": "Asymmetry and atypical pigment network noted. Punch biopsy recommended within 10 days.",
        },
    )
    assert review_resp.status_code == 200
    review_data = review_resp.json()
    assert review_data["review_status"] == "biopsy_recommended"
    assert "Punch biopsy recommended" in review_data["doctor_review_notes"]

    # 8. Patient 1 checks case and sees the doctor's review notes and decision
    p1_check = client.get(f"/api/v1/cases/{case_id}", headers=p1_headers)
    assert p1_check.status_code == 200
    p1_case_data = p1_check.json()
    assert p1_case_data["review_status"] == "biopsy_recommended"
    assert "Punch biopsy recommended" in p1_case_data["doctor_review_notes"]
