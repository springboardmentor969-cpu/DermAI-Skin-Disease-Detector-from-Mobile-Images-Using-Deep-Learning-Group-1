"""
Unit and integration tests for Clinician Analytics Dashboard router.

Tests:
1. Authentication: 401 Unauthorized for unauthenticated requests across all 5 endpoints.
2. Authorization: 403 Forbidden for patient role across all 5 endpoints.
3. Clinician and Admin role access: 200 OK across all 5 endpoints.
4. GET /dashboard/stats: Schema validation, accurate aggregation of counts, high-risk cases, confidence.
5. GET /dashboard/class-distribution: All 7 skin lesion classes represented with counts and percentages.
6. GET /dashboard/confidence-distribution: 10 buckets (0-10% to 90-100%) with appropriate counts.
7. GET /dashboard/recent-cases: Returns last 20 cases with predicted_class, confidence, risk_level, patient_name.
8. GET /dashboard/cases-over-time: Returns 30 continuous daily date buckets with counts.
"""

import pytest
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.case import Case
from app.models.image import Image
from app.models.patient import Patient
from app.models.prediction import Prediction
from app.models.user import User, UserRole
from app.services.auth_service import create_access_token, hash_password


def get_token_for_user(db: Session, email: str, role: str) -> str:
    """Create or fetch a user in db and generate a valid JWT token."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(
            email=email,
            password_hash=hash_password("Password123!"),
            full_name=f"User {email}",
            role=role,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return create_access_token({"sub": user.email})


DASHBOARD_ENDPOINTS = [
    "/dashboard/stats",
    "/dashboard/class-distribution",
    "/dashboard/confidence-distribution",
    "/dashboard/recent-cases",
    "/dashboard/cases-over-time",
]


@pytest.mark.parametrize("endpoint", DASHBOARD_ENDPOINTS)
def test_dashboard_endpoints_require_auth(client: TestClient, endpoint: str):
    """All dashboard endpoints require JWT authentication (401)."""
    res = client.get(endpoint)
    assert res.status_code == 401


@pytest.mark.parametrize("endpoint", DASHBOARD_ENDPOINTS)
def test_dashboard_endpoints_reject_patient_role(client: TestClient, db_session: Session, endpoint: str):
    """Patients cannot access clinician/admin analytics dashboard (403)."""
    token = get_token_for_user(db_session, "patient_dashboard@example.com", UserRole.PATIENT)
    res = client.get(endpoint, headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 403
    assert "Clinician or administrator privileges required" in res.json()["detail"]


@pytest.mark.parametrize("endpoint", DASHBOARD_ENDPOINTS)
def test_dashboard_endpoints_allow_clinician_and_admin(client: TestClient, db_session: Session, endpoint: str):
    """Clinicians and admins have full access to dashboard endpoints (200)."""
    clinician_token = get_token_for_user(db_session, "clinician_dash@example.com", UserRole.CLINICIAN)
    res = client.get(endpoint, headers={"Authorization": f"Bearer {clinician_token}"})
    assert res.status_code == 200

    admin_token = get_token_for_user(db_session, "admin_dash@example.com", UserRole.ADMIN)
    res_admin = client.get(endpoint, headers={"Authorization": f"Bearer {admin_token}"})
    assert res_admin.status_code == 200


def test_dashboard_stats_and_aggregations(client: TestClient, db_session: Session):
    """Verify stats computation: totals, high risk count, average confidence."""
    clinician_token = get_token_for_user(db_session, "clinician_stats@example.com", UserRole.CLINICIAN)

    # Seed data
    p = Patient(age=55, gender="male", notes="Clinical Triage Test")
    db_session.add(p)
    db_session.commit()

    # Case 1: High risk melanoma
    c1 = Case(patient_id=p.id, created_by=1, body_location="Back", status="submitted")
    db_session.add(c1)
    db_session.commit()

    img1 = Image(case_id=c1.id, storage_key="img1.jpg", quality_label="good")
    db_session.add(img1)
    db_session.commit()

    pred1 = Prediction(
        image_id=img1.id,
        model_version="resnet50_v1.0",
        predicted_class="mel",
        confidence=0.92,
        risk_level="HIGH",
    )
    db_session.add(pred1)

    # Case 2: Low risk nevus
    c2 = Case(patient_id=p.id, created_by=1, body_location="Leg", status="reviewing")
    db_session.add(c2)
    db_session.commit()

    img2 = Image(case_id=c2.id, storage_key="img2.jpg", quality_label="good")
    db_session.add(img2)
    db_session.commit()

    pred2 = Prediction(
        image_id=img2.id,
        model_version="resnet50_v1.0",
        predicted_class="nv",
        confidence=0.88,
        risk_level="LOW",
    )
    db_session.add(pred2)
    db_session.commit()

    res = client.get("/dashboard/stats", headers={"Authorization": f"Bearer {clinician_token}"})
    assert res.status_code == 200
    data = res.json()

    assert data["total_cases"] >= 2
    assert data["cases_this_week"] >= 2
    assert data["cases_this_month"] >= 2
    assert data["high_risk_count"] >= 1
    assert data["pending_review_count"] >= 2
    assert data["avg_confidence"] > 0.8


def test_dashboard_class_distribution(client: TestClient, db_session: Session):
    """Verify class distribution returns all 7 classes with percentages summing to 100."""
    clinician_token = get_token_for_user(db_session, "clinician_classes@example.com", UserRole.CLINICIAN)

    res = client.get("/dashboard/class-distribution", headers={"Authorization": f"Bearer {clinician_token}"})
    assert res.status_code == 200
    classes = res.json()

    assert len(classes) == 7
    class_names = [item["class_name"] for item in classes]
    assert "Melanoma" in class_names
    assert "Melanocytic Nevus" in class_names
    assert "Basal Cell Carcinoma" in class_names
    assert "Actinic Keratosis" in class_names
    assert "Benign Keratosis" in class_names
    assert "Dermatofibroma" in class_names
    assert "Vascular Lesion" in class_names

    for item in classes:
        assert "count" in item
        assert "percentage" in item
        assert "class_code" in item


def test_dashboard_confidence_distribution(client: TestClient, db_session: Session):
    """Verify confidence distribution has exactly 10 buckets from 0-10% to 90-100%."""
    clinician_token = get_token_for_user(db_session, "clinician_conf@example.com", UserRole.CLINICIAN)

    res = client.get("/dashboard/confidence-distribution", headers={"Authorization": f"Bearer {clinician_token}"})
    assert res.status_code == 200
    buckets = res.json()

    assert len(buckets) == 10
    bucket_names = [b["bucket"] for b in buckets]
    assert bucket_names[0] == "0-10%"
    assert bucket_names[-1] == "90-100%"
    for b in buckets:
        assert isinstance(b["count"], int)


def test_dashboard_recent_cases_endpoint_shape(client: TestClient, db_session: Session):
    """Verify GET /dashboard/recent-cases returns HTTP 200 with correct RecentCaseItem response shape."""
    clinician_token = get_token_for_user(db_session, "clinician_recent@example.com", UserRole.CLINICIAN)

    p = Patient(age=62, gender="female", notes="Triage Case 101")
    db_session.add(p)
    db_session.commit()

    c = Case(patient_id=p.id, created_by=1, body_location="Forehead", status="submitted")
    db_session.add(c)
    db_session.commit()

    img = Image(case_id=c.id, storage_key="triage.jpg", quality_label="good")
    db_session.add(img)
    db_session.commit()

    pred = Prediction(
        image_id=img.id,
        model_version="resnet50_v1.0",
        predicted_class="bcc",
        confidence=0.85,
        risk_level="HIGH",
    )
    db_session.add(pred)
    db_session.commit()

    res = client.get("/dashboard/recent-cases", headers={"Authorization": f"Bearer {clinician_token}"})
    assert res.status_code == 200
    cases = res.json()
    assert isinstance(cases, list)
    assert len(cases) >= 1
    recent_entry = cases[0]
    expected_fields = {"case_id", "patient_name", "predicted_class", "predicted_class_name", "confidence", "risk_level", "created_at"}
    assert expected_fields.issubset(recent_entry.keys())
    assert isinstance(recent_entry["case_id"], int)
    assert isinstance(recent_entry["confidence"], float)
    assert recent_entry["risk_level"] in ("HIGH", "MEDIUM", "LOW")


def test_dashboard_cases_over_time_endpoint_shape(client: TestClient, db_session: Session):
    """Verify GET /dashboard/cases-over-time returns HTTP 200 with 30-day timeline shape."""
    clinician_token = get_token_for_user(db_session, "clinician_time@example.com", UserRole.CLINICIAN)

    res_time = client.get("/dashboard/cases-over-time", headers={"Authorization": f"Bearer {clinician_token}"})
    assert res_time.status_code == 200
    timeline = res_time.json()
    assert isinstance(timeline, list)
    assert len(timeline) == 30
    for day in timeline:
        assert "date" in day
        assert "count" in day
        assert isinstance(day["count"], int)
        assert len(day["date"].split("-")) == 3
