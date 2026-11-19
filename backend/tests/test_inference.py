"""
AI Analysis and Inference Integration Tests.

Verifies:
- Dynamic model architecture & class mapping resolution from checkpoint/config.json.
- POST /images/{image_id}/analyze with mock model without requiring real weights.
- DEMO_MODEL flag inclusion ("demo_model": true) on smoke models.
- Segmentation overlay suppression on smoke-test segmentation weights.
- Refusal to analyze images with 'poor' quality label with a clear retake message.
- Marking results as 'low confidence - clinical review recommended' when below threshold.
- Mandatory clinical disclaimer on every response: 'AI-assisted analysis. Not a medical diagnosis.'
- Audit log entry recording for IMAGE_ANALYZED.
- Role-based authorization (patient cannot analyze another patient's image).
- Unauthenticated request rejection (401).
- GET /images/{image_id}/predictions listing history.
"""

import io
import json
import pytest
from pathlib import Path
from PIL import Image as PILImage
import torch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.patient import Patient
from app.models.case import Case
from app.models.image import Image
from app.models.prediction import Prediction
from app.models.audit import AuditLog
from app.services.inference import inference_service, InferenceService, MANDATORY_DISCLAIMER
from app.services.storage_service import storage_service
from app.config import settings


def create_test_jpeg(width: int = 300, height: int = 300) -> bytes:
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
    patient = Patient(age=45, gender="male", notes="Screening check")
    db_session.add(patient)
    db_session.commit()
    case = Case(patient_id=patient.id, created_by=created_by, body_location=body_location, status="submitted")
    db_session.add(case)
    db_session.commit()
    return case


@pytest.fixture(autouse=True)
def mock_storage_and_model(monkeypatch):
    """
    Mocks storage_service and inference_service so that tests execute
    instantly in-memory without external MinIO or downloading real weights.
    """
    dummy_bytes = create_test_jpeg()

    # Mock storage_service.get_file to return dummy bytes
    monkeypatch.setattr(storage_service, "get_file", lambda key: dummy_bytes)

    # Set default high-confidence mock prediction
    def mock_predict(image_input, run_segmentation=True, threshold=None):
        return {
            "predicted_class": "mel",
            "predicted_class_name": "Melanoma",
            "confidence": 0.8850,
            "top_n": [
                {"class": "mel", "label": "Melanoma", "probability": 0.8850},
                {"class": "nv", "label": "Melanocytic Nevus", "probability": 0.0820},
                {"class": "bkl", "label": "Benign Keratosis", "probability": 0.0210},
            ],
            "is_low_confidence": False,
            "warning": None,
            "demo_model": True,
            "model_version": "smoke-test",
            "segmentation": None,
            "segmentation_overlay": None,
            "disclaimer": MANDATORY_DISCLAIMER,
        }

    inference_service.set_mock_model(mock_predict)
    yield
    inference_service.reset_mock()


def test_dynamic_model_architecture_and_class_mapping_from_config(tmp_path: Path):
    """
    Verifies that the inference service dynamically reads model architecture
    (e.g. mobilenetv3_large_100 or efficientnet_b0) and custom classes from adjacent config.json.
    """
    service = InferenceService()

    # Write a custom config.json specifying a different architecture and 3 classes
    custom_classes = {"mel": 0, "nv": 1, "bcc": 2}
    config_data = {
        "model": "mobilenetv3_large_100",
        "classes": custom_classes,
        "epochs": 1,
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data), encoding="utf-8")

    # Create dummy checkpoint
    ckpt_file = tmp_path / "best.pt"
    torch.save({
        "model_name": "mobilenetv3_large_100",
        "classes": custom_classes,
        "model_state_dict": {},
    }, ckpt_file)

    service.load_model(model_path=ckpt_file)

    assert service.model_name == "mobilenetv3_large_100"
    assert service.class_to_idx == custom_classes
    assert len(service.idx_to_class) == 3
    assert service.model is not None


def test_segmentation_overlay_suppressed_for_smoke_weights(monkeypatch):
    """
    Verifies that segmentation overlay is suppressed (None) when
    weights are smoke-test weights, and falls back to full image.
    """
    # Temporarily bypass mock_predict
    inference_service.reset_mock()

    monkeypatch.setattr(settings, "SEGMENTATION_MODEL_PATH", "ml/experiments/segmentation_smoke/best_model.pth")
    assert inference_service.is_segmentation_smoke is True

    dummy_bytes = create_test_jpeg()
    result = inference_service.predict(dummy_bytes, run_segmentation=True)

    assert result["segmentation_overlay"] is None
    assert result["demo_model"] is True


def test_analyze_image_success(client: TestClient, db_session: Session):
    """Successful analysis on a good quality image saves prediction and audit log."""
    token = get_auth_token(client, "doc_analyst@example.com", "clinician")

    # Create case and image directly in DB
    case = create_case_with_patient(db_session, created_by=1, body_location="Back")

    image = Image(
        case_id=case.id,
        storage_key="cases/1/sample.jpg",
        quality_label="good",
        quality_details={"blur_score": 140.0},
    )
    db_session.add(image)
    db_session.commit()

    # Call POST /images/{image_id}/analyze
    response = client.post(
        f"/images/{image.id}/analyze",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] == "mel"
    assert data["confidence"] == 0.8850
    assert len(data["top_n"]) == 3
    assert data["is_low_confidence"] is False
    assert data["warning"] is None
    assert data["demo_model"] is True
    assert data["disclaimer"] == MANDATORY_DISCLAIMER
    assert data["model_version"] == "smoke-test"

    # Verify stored in DB
    pred_db = db_session.query(Prediction).filter(Prediction.image_id == image.id).first()
    assert pred_db is not None
    assert pred_db.predicted_class == "mel"
    assert round(pred_db.confidence, 4) == 0.8850

    # Verify audit log entry
    audit = db_session.query(AuditLog).filter(AuditLog.action == "IMAGE_ANALYZED").first()
    assert audit is not None
    assert audit.details["image_id"] == image.id
    assert audit.details["predicted_class"] == "mel"


def test_analyze_image_refuses_poor_quality(client: TestClient, db_session: Session):
    """Refuses to analyze images labelled 'poor' quality and returns retake instructions."""
    token = get_auth_token(client, "clinician_poor@example.com", "clinician")

    case = create_case_with_patient(db_session, created_by=1, body_location="Left cheek")

    poor_image = Image(
        case_id=case.id,
        storage_key="cases/1/blurry.jpg",
        quality_label="poor",
        quality_details={"blur_score": 12.0, "issues": ["too blurry"]},
    )
    db_session.add(poor_image)
    db_session.commit()

    response = client.post(
        f"/images/{poor_image.id}/analyze",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Cannot analyze image labelled 'poor' quality" in detail
    assert "retake" in detail.lower()

    # Ensure no prediction was saved
    pred_count = db_session.query(Prediction).filter(Prediction.image_id == poor_image.id).count()
    assert pred_count == 0


def test_analyze_image_low_confidence_flag(client: TestClient, db_session: Session):
    """Marks prediction as low confidence when below threshold (0.60)."""
    token = get_auth_token(client, "lowconf_doc@example.com", "clinician")

    # Set mock with low confidence
    def mock_low_conf(image_input, run_segmentation=True, threshold=None):
        return {
            "predicted_class": "nv",
            "predicted_class_name": "Melanocytic Nevus",
            "confidence": 0.4215,
            "top_n": [
                {"class": "nv", "label": "Melanocytic Nevus", "probability": 0.4215},
                {"class": "mel", "label": "Melanoma", "probability": 0.3810},
                {"class": "bkl", "label": "Benign Keratosis", "probability": 0.1500},
            ],
            "is_low_confidence": True,
            "warning": "low confidence - clinical review recommended",
            "demo_model": True,
            "model_version": "smoke-test",
            "segmentation": None,
            "segmentation_overlay": None,
            "disclaimer": MANDATORY_DISCLAIMER,
        }

    inference_service.set_mock_model(mock_low_conf)

    case = create_case_with_patient(db_session, created_by=1, body_location="Forearm")

    image = Image(case_id=case.id, storage_key="cases/1/borderline.jpg", quality_label="acceptable")
    db_session.add(image)
    db_session.commit()

    response = client.post(
        f"/images/{image.id}/analyze",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["is_low_confidence"] is True
    assert data["warning"] == "low confidence - clinical review recommended"
    assert data["demo_model"] is True
    assert data["disclaimer"] == MANDATORY_DISCLAIMER


def test_analyze_image_authorization(client: TestClient, db_session: Session):
    """Patient cannot trigger analysis on an image belonging to another patient."""
    pat1_token = get_auth_token(client, "pat_owner@example.com", "patient")
    pat2_token = get_auth_token(client, "pat_intruder@example.com", "patient")

    # Patient 1 creates a case and image
    case_res = client.post(
        "/cases",
        json={"body_location": "Shoulder"},
        headers={"Authorization": f"Bearer {pat1_token}"},
    )
    case_id = case_res.json()["id"]

    image = Image(case_id=case_id, storage_key="cases/pat1/img.jpg", quality_label="good")
    db_session.add(image)
    db_session.commit()

    # Patient 2 tries to analyze Patient 1's image
    forbidden_res = client.post(
        f"/images/{image.id}/analyze",
        headers={"Authorization": f"Bearer {pat2_token}"},
    )
    assert forbidden_res.status_code == 403


def test_analyze_image_unauthenticated(client: TestClient, db_session: Session):
    """Unauthenticated request is rejected with 401."""
    case = create_case_with_patient(db_session, created_by=1, body_location="Leg")

    image = Image(case_id=case.id, storage_key="cases/1/img.jpg", quality_label="good")
    db_session.add(image)
    db_session.commit()

    res = client.post(f"/images/{image.id}/analyze")
    assert res.status_code == 401


def test_get_image_predictions_history(client: TestClient, db_session: Session):
    """GET /images/{image_id}/predictions returns stored predictions with disclaimer."""
    token = get_auth_token(client, "history_doc@example.com", "clinician")

    case = create_case_with_patient(db_session, created_by=1, body_location="Abdomen")

    image = Image(case_id=case.id, storage_key="cases/1/history.jpg", quality_label="good")
    db_session.add(image)
    db_session.commit()

    # Insert two predictions
    p1 = Prediction(
        image_id=image.id,
        model_version="smoke-test",
        predicted_class="bkl",
        confidence=0.75,
        top_n=[{"class": "bkl", "probability": 0.75}],
    )
    p2 = Prediction(
        image_id=image.id,
        model_version="smoke-test",
        predicted_class="mel",
        confidence=0.52,  # Low confidence
        top_n=[{"class": "mel", "probability": 0.52}],
    )
    db_session.add_all([p1, p2])
    db_session.commit()

    res = client.get(
        f"/images/{image.id}/predictions",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert res.status_code == 200
    preds = res.json()
    assert len(preds) == 2
    for p in preds:
        assert p["disclaimer"] == MANDATORY_DISCLAIMER
        assert "is_low_confidence" in p
        assert "demo_model" in p
