"""
Unit and integration tests for clinician review features:
1. POST /images/{image_id}/review (clinician role enforcement, diagnosis confirmation, override)
2. GET /images/{image_id}/review (fetch review)
3. GET /clinician/pending (pending triage list)
4. GET /clinician/stats (review metrics, AI accuracy, weekly overrides)
5. GET /clinician/cases (case filtering)
"""

import io
from PIL import Image as PILImage
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.case import Case
from app.models.image import Image
from app.models.patient import Patient
from app.models.prediction import Prediction
from app.models.clinical_review import ClinicalReview
from app.models.user import User, UserRole
from app.services.auth_service import hash_password, create_access_token


@pytest.fixture
def test_setup(db_session: Session):
    """Sets up a clinician, a patient, a case, an image, and a prediction."""
    # Clinician user
    clinician = User(
        email="dr.derm@example.com",
        password_hash=hash_password("DoctorPass123!"),
        full_name="Dr. Dermatologist",
        role=UserRole.CLINICIAN,
    )
    # Patient user
    patient_user = User(
        email="patient_review@example.com",
        password_hash=hash_password("PatientPass123!"),
        full_name="John Doe",
        role=UserRole.PATIENT,
    )
    db_session.add(clinician)
    db_session.add(patient_user)
    db_session.commit()
    db_session.refresh(clinician)
    db_session.refresh(patient_user)

    # Patient record
    patient = Patient(age=45, gender="male", notes="Back lesion")
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    # Case
    test_case = Case(
        patient_id=patient.id,
        created_by=clinician.id,
        body_location="Upper Back",
        status="submitted",
    )
    db_session.add(test_case)
    db_session.commit()
    db_session.refresh(test_case)

    # Image
    image = Image(
        case_id=test_case.id,
        storage_key=f"cases/{test_case.id}/sample.jpg",
        source="camera",
        quality_label="good",
    )
    db_session.add(image)
    db_session.commit()
    db_session.refresh(image)

    # Prediction
    pred = Prediction(
        image_id=image.id,
        model_version="resnet50_v1",
        predicted_class="mel",
        confidence=0.885,
        risk_level="HIGH",
        top_n=[{"class": "mel", "probability": 0.885}, {"class": "nv", "probability": 0.08}],
    )
    db_session.add(pred)
    db_session.commit()
    db_session.refresh(pred)

    clinician_token = create_access_token(
        data={"sub": clinician.email, "user_id": clinician.id, "role": "clinician"}
    )
    patient_token = create_access_token(
        data={"sub": patient_user.email, "user_id": patient_user.id, "role": "patient"}
    )

    return {
        "clinician": clinician,
        "patient_user": patient_user,
        "case": test_case,
        "image": image,
        "prediction": pred,
        "clinician_token": clinician_token,
        "patient_token": patient_token,
    }


def test_review_requires_clinician_role(client: TestClient, test_setup):
    """Patient attempting to submit a clinical review receives HTTP 403 Forbidden."""
    image_id = test_setup["image"].id
    patient_headers = {"Authorization": f"Bearer {test_setup['patient_token']}"}

    res = client.post(
        f"/images/{image_id}/review",
        json={"notes": "Unauthorized review attempt", "confirmed_class": "mel", "overridden": False},
        headers=patient_headers,
    )
    assert res.status_code == 403
    assert "insufficient permissions" in res.json()["detail"].lower()


def test_clinician_submit_and_get_review(client: TestClient, test_setup, db_session: Session):
    """Clinician submits diagnostic review and fetches it successfully."""
    image_id = test_setup["image"].id
    headers = {"Authorization": f"Bearer {test_setup['clinician_token']}"}

    # 1. Submit review
    review_data = {
        "notes": "Classic atypical network with asymmetry. Confirmed superficial spreading melanoma.",
        "confirmed_class": "mel",
        "overridden": False,
    }
    submit_res = client.post(f"/images/{image_id}/review", json=review_data, headers=headers)
    assert submit_res.status_code == 200
    res_json = submit_res.json()
    assert res_json["image_id"] == image_id
    assert res_json["confirmed_class"] == "mel"
    assert res_json["confirmed_class_name"] == "Melanoma"
    assert res_json["overridden"] is False
    assert res_json["notes"] == review_data["notes"]

    # 2. Verify record in clinical_reviews table
    db_rev = db_session.query(ClinicalReview).filter(ClinicalReview.image_id == image_id).first()
    assert db_rev is not None
    assert db_rev.confirmed_class == "mel"
    assert db_rev.overridden is False

    # 3. GET /images/{image_id}/review
    get_res = client.get(f"/images/{image_id}/review", headers=headers)
    assert get_res.status_code == 200
    assert get_res.json()["id"] == res_json["id"]


def test_clinician_override_ai_prediction(client: TestClient, test_setup, db_session: Session):
    """Clinician overrides AI prediction and flags discrepancy."""
    image_id = test_setup["image"].id
    headers = {"Authorization": f"Bearer {test_setup['clinician_token']}"}

    override_data = {
        "notes": "AI flagged Melanoma, but dermoscopy reveals clear arborizing vessels typical of BCC.",
        "confirmed_class": "bcc",
        "overridden": True,
    }
    res = client.post(f"/images/{image_id}/review", json=override_data, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["confirmed_class"] == "bcc"
    assert data["overridden"] is True

    # Check database
    db_rev = db_session.query(ClinicalReview).filter(ClinicalReview.image_id == image_id).first()
    assert db_rev.overridden is True
    assert db_rev.confirmed_class == "bcc"


def test_clinician_pending_and_stats_endpoints(client: TestClient, test_setup, db_session: Session):
    """Validates /clinician/pending and /clinician/stats endpoints."""
    headers = {"Authorization": f"Bearer {test_setup['clinician_token']}"}

    # Clean reviews
    db_session.query(ClinicalReview).delete()
    db_session.commit()

    # Before review: image should appear in pending list
    pending_res = client.get("/clinician/pending", headers=headers)
    assert pending_res.status_code == 200
    pending_items = pending_res.json()
    assert any(item["image_id"] == test_setup["image"].id for item in pending_items)

    # Check stats before review
    stats_res = client.get("/clinician/stats", headers=headers)
    assert stats_res.status_code == 200
    stats_before = stats_res.json()
    assert stats_before["pending_review"] >= 1

    # Now submit review
    client.post(
        f"/images/{test_setup['image'].id}/review",
        json={"notes": "Reviewed", "confirmed_class": "mel", "overridden": False},
        headers=headers,
    )

    # After review: pending should no longer list this image
    pending_after = client.get("/clinician/pending", headers=headers).json()
    assert not any(item["image_id"] == test_setup["image"].id for item in pending_after)

    # Check stats after review
    stats_after = client.get("/clinician/stats", headers=headers).json()
    assert stats_after["total_cases_reviewed"] >= 1
    assert stats_after["ai_accuracy"] == 100.0


def test_clinician_cases_filter(client: TestClient, test_setup):
    """Tests /clinician/cases with filter tabs."""
    headers = {"Authorization": f"Bearer {test_setup['clinician_token']}"}

    res_all = client.get("/clinician/cases?filter=all", headers=headers)
    assert res_all.status_code == 200
    assert isinstance(res_all.json(), list)

    res_pending = client.get("/clinician/cases?filter=pending", headers=headers)
    assert res_pending.status_code == 200

    res_reviewed = client.get("/clinician/cases?filter=reviewed", headers=headers)
    assert res_reviewed.status_code == 200
