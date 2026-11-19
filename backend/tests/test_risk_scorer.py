"""
Unit and integration tests for DermAI 3-tier risk stratification engine.

Tests:
1. compute_risk logic:
   - HIGH: mel at any conf, bcc/akiec at conf >= 0.5
   - MEDIUM: bcc/akiec at conf < 0.5, bkl at conf < 0.4, any class at conf < 0.45
   - LOW: nv/df/vasc at conf >= 0.5, bkl at conf >= 0.4
2. InferenceService includes all 4 risk fields in prediction dict.
3. Prediction DB model and alembic schema contain risk_level column.
4. POST /images/{id}/analyze returns risk fields and stores risk_level in database.
"""

import io
import pytest
from pathlib import Path
from PIL import Image as PILImage
import torch
import timm
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.case import Case
from app.models.image import Image
from app.models.patient import Patient
from app.models.prediction import Prediction
from app.services.inference import inference_service, InferenceService
from app.services.risk_scorer import compute_risk
from app.services.storage_service import storage_service


def create_test_jpeg(width: int = 224, height: int = 224) -> bytes:
    """Generate in-memory RGB JPEG image bytes."""
    buf = io.BytesIO()
    img = PILImage.new("RGB", (width, height), color=(180, 100, 80))
    img.save(buf, format="JPEG")
    return buf.getvalue()


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


def create_case_with_patient(db_session: Session, created_by: int = 1, body_location: str = "Back") -> Case:
    """Helper to create patient and associated case satisfying foreign key constraints."""
    patient = Patient(age=45, gender="male", notes="Risk test patient")
    db_session.add(patient)
    db_session.commit()
    case = Case(patient_id=patient.id, created_by=created_by, body_location=body_location, status="submitted")
    db_session.add(case)
    db_session.commit()
    return case


def test_compute_risk_high_tier():
    """Verifies HIGH tier rules (urgency: 1 day, color: red)."""
    # mel at any confidence
    r_mel_low = compute_risk("mel", 0.15)
    assert r_mel_low["risk_level"] == "HIGH"
    assert r_mel_low["risk_color"] == "red"
    assert r_mel_low["urgency_days"] == 1

    r_mel_high = compute_risk("mel", 0.95)
    assert r_mel_high["risk_level"] == "HIGH"
    assert r_mel_high["risk_color"] == "red"
    assert r_mel_high["urgency_days"] == 1

    # bcc at confidence >= 0.5
    r_bcc = compute_risk("bcc", 0.50)
    assert r_bcc["risk_level"] == "HIGH"
    assert r_bcc["risk_color"] == "red"
    assert r_bcc["urgency_days"] == 1

    # akiec at confidence >= 0.5
    r_akiec = compute_risk("akiec", 0.72)
    assert r_akiec["risk_level"] == "HIGH"
    assert r_akiec["risk_color"] == "red"
    assert r_akiec["urgency_days"] == 1


def test_compute_risk_medium_tier():
    """Verifies MEDIUM tier rules (urgency: 14 days, color: amber)."""
    # bcc or akiec at confidence < 0.5
    r_bcc_low = compute_risk("bcc", 0.48)
    assert r_bcc_low["risk_level"] == "MEDIUM"
    assert r_bcc_low["risk_color"] == "amber"
    assert r_bcc_low["urgency_days"] == 14

    r_akiec_low = compute_risk("akiec", 0.46)
    assert r_akiec_low["risk_level"] == "MEDIUM"
    assert r_akiec_low["risk_color"] == "amber"
    assert r_akiec_low["urgency_days"] == 14

    # bkl at confidence < 0.4
    r_bkl_low = compute_risk("bkl", 0.38)
    assert r_bkl_low["risk_level"] == "MEDIUM"
    assert r_bkl_low["risk_color"] == "amber"
    assert r_bkl_low["urgency_days"] == 14

    # Any class where confidence < 0.45
    r_nv_uncertain = compute_risk("nv", 0.42)
    assert r_nv_uncertain["risk_level"] == "MEDIUM"
    assert r_nv_uncertain["risk_color"] == "amber"
    assert r_nv_uncertain["urgency_days"] == 14


def test_compute_risk_low_tier():
    """Verifies LOW tier rules (urgency: 90 days, color: green)."""
    # nv at confidence >= 0.5
    r_nv = compute_risk("nv", 0.50)
    assert r_nv["risk_level"] == "LOW"
    assert r_nv["risk_color"] == "green"
    assert r_nv["urgency_days"] == 90

    # df at confidence >= 0.5
    r_df = compute_risk("df", 0.88)
    assert r_df["risk_level"] == "LOW"
    assert r_df["risk_color"] == "green"
    assert r_df["urgency_days"] == 90

    # vasc at confidence >= 0.5
    r_vasc = compute_risk("vasc", 0.65)
    assert r_vasc["risk_level"] == "LOW"
    assert r_vasc["risk_color"] == "green"
    assert r_vasc["urgency_days"] == 90

    # bkl at confidence >= 0.4
    r_bkl = compute_risk("bkl", 0.40)
    assert r_bkl["risk_level"] == "LOW"
    assert r_bkl["risk_color"] == "green"
    assert r_bkl["urgency_days"] == 90


def test_all_risk_tier_boundaries():
    """
    Exhaustive boundary testing:
    - mel always HIGH (at 0.05, 0.40, 0.499, 0.50, 0.99)
    - nv + high confidence (>= 0.50) = LOW, low confidence (< 0.50) = MEDIUM
    - bcc >= 0.50 = HIGH, bcc below 0.50 (< 0.50) = MEDIUM
    - akiec >= 0.50 = HIGH, akiec below 0.50 (< 0.50) = MEDIUM
    - df >= 0.50 = LOW, df < 0.50 = MEDIUM
    - vasc >= 0.50 = LOW, vasc < 0.50 = MEDIUM
    - bkl >= 0.40 = LOW, bkl < 0.40 = MEDIUM
    - Any class with confidence < 0.45 = MEDIUM (if not mel or already HIGH)
    """
    # 1. Melanoma (mel) is always HIGH regardless of confidence
    for c in [0.01, 0.10, 0.35, 0.449, 0.499, 0.50, 0.501, 0.95]:
        r = compute_risk("mel", c)
        assert r["risk_level"] == "HIGH", f"mel at {c} should be HIGH"
        assert r["urgency_days"] == 1
        assert r["risk_color"] == "red"

    # 2. bcc boundaries: >= 0.50 is HIGH, < 0.50 is MEDIUM
    assert compute_risk("bcc", 0.500)["risk_level"] == "HIGH"
    assert compute_risk("bcc", 0.501)["risk_level"] == "HIGH"
    assert compute_risk("bcc", 0.499)["risk_level"] == "MEDIUM"
    assert compute_risk("bcc", 0.300)["risk_level"] == "MEDIUM"
    assert compute_risk("bcc", 0.499)["urgency_days"] == 14

    # 3. akiec boundaries: >= 0.50 is HIGH, < 0.50 is MEDIUM
    assert compute_risk("akiec", 0.500)["risk_level"] == "HIGH"
    assert compute_risk("akiec", 0.501)["risk_level"] == "HIGH"
    assert compute_risk("akiec", 0.499)["risk_level"] == "MEDIUM"
    assert compute_risk("akiec", 0.250)["risk_level"] == "MEDIUM"
    assert compute_risk("akiec", 0.499)["urgency_days"] == 14

    # 4. nv boundaries: >= 0.50 is LOW, < 0.50 is MEDIUM
    assert compute_risk("nv", 0.500)["risk_level"] == "LOW"
    assert compute_risk("nv", 0.900)["risk_level"] == "LOW"
    assert compute_risk("nv", 0.499)["risk_level"] == "MEDIUM"
    assert compute_risk("nv", 0.400)["risk_level"] == "MEDIUM"
    assert compute_risk("nv", 0.500)["urgency_days"] == 90
    assert compute_risk("nv", 0.499)["urgency_days"] == 14

    # 5. df boundaries: >= 0.50 is LOW, < 0.50 is MEDIUM
    assert compute_risk("df", 0.500)["risk_level"] == "LOW"
    assert compute_risk("df", 0.850)["risk_level"] == "LOW"
    assert compute_risk("df", 0.499)["risk_level"] == "MEDIUM"
    assert compute_risk("df", 0.300)["risk_level"] == "MEDIUM"

    # 6. vasc boundaries: >= 0.50 is LOW, < 0.50 is MEDIUM
    assert compute_risk("vasc", 0.500)["risk_level"] == "LOW"
    assert compute_risk("vasc", 0.750)["risk_level"] == "LOW"
    assert compute_risk("vasc", 0.499)["risk_level"] == "MEDIUM"
    assert compute_risk("vasc", 0.200)["risk_level"] == "MEDIUM"

    # 7. bkl boundaries: >= 0.40 is LOW, < 0.40 is MEDIUM
    assert compute_risk("bkl", 0.400)["risk_level"] == "LOW"
    assert compute_risk("bkl", 0.600)["risk_level"] == "LOW"
    assert compute_risk("bkl", 0.399)["risk_level"] == "MEDIUM"
    assert compute_risk("bkl", 0.200)["risk_level"] == "MEDIUM"


def test_inference_service_includes_risk_fields(tmp_path: Path):
    """Verifies that InferenceService includes all 4 risk fields."""
    model_file = tmp_path / "risk_m.pt"
    m = timm.create_model("resnet18", pretrained=False, num_classes=7)
    torch.save({
        "model_name": "resnet18",
        "model_state_dict": m.state_dict(),
    }, model_file)

    service = InferenceService()
    service.reset_mock()
    service.load_model(model_file)

    dummy_img = PILImage.new("RGB", (224, 224), color=(150, 100, 90))
    res = service.predict(dummy_img, run_segmentation=False)

    assert "risk_level" in res
    assert res["risk_level"] in ("HIGH", "MEDIUM", "LOW")
    assert "risk_color" in res
    assert res["risk_color"] in ("red", "amber", "green")
    assert "risk_message" in res
    assert isinstance(res["risk_message"], str)
    assert "urgency_days" in res
    assert res["urgency_days"] in (1, 14, 90)


def test_analyze_image_api_returns_and_stores_risk(client: TestClient, db_session: Session, monkeypatch):
    """
    Verifies full API flow:
    - POST /images/{id}/analyze returns all 4 risk fields
    - Persists risk_level column in predictions table
    """
    clinician_token = get_auth_token(client, "risk_doc@example.com", "clinician")
    case = create_case_with_patient(db_session, created_by=1, body_location="Back")
    image = Image(case_id=case.id, storage_key="cases/1/back.jpg", quality_label="good")
    db_session.add(image)
    db_session.commit()

    dummy_bytes = create_test_jpeg()
    monkeypatch.setattr(storage_service, "get_file", lambda key: dummy_bytes)

    # Use mock predicting melanoma (HIGH risk)
    def mock_predict(img_input, run_segmentation=True, threshold=None, image_id=None):
        r = compute_risk("mel", 0.91)
        return {
            "predicted_class": "mel",
            "predicted_class_name": "Melanoma",
            "confidence": 0.91,
            "risk_level": r["risk_level"],
            "risk_color": r["risk_color"],
            "risk_message": r["risk_message"],
            "urgency_days": r["urgency_days"],
            "top_n": [{"class": "mel", "label": "Melanoma", "probability": 0.91}],
            "is_low_confidence": False,
            "warning": None,
            "demo_model": False,
            "model_version": "resnet50_v1.0",
            "segmentation": None,
            "segmentation_overlay": None,
            "gradcam_key": None,
            "disclaimer": "AI-assisted analysis. Not a medical diagnosis.",
        }

    inference_service.set_mock_model(mock_predict)

    res = client.post(
        f"/images/{image.id}/analyze",
        headers={"Authorization": f"Bearer {clinician_token}"},
    )

    assert res.status_code == 200
    data = res.json()
    assert data["risk_level"] == "HIGH"
    assert data["risk_color"] == "red"
    assert data["urgency_days"] == 1
    assert "Urgent dermatologist review" in data["risk_message"]

    # Verify risk_level was persisted in the Prediction database record
    db_pred = db_session.query(Prediction).filter(Prediction.image_id == image.id).first()
    assert db_pred is not None
    assert db_pred.risk_level == "HIGH"

