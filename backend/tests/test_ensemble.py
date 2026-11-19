"""
Unit and integration tests for DermAI Multi-Model Ensemble Inference Service.

Tests:
1. ensemble_predict loads checkpoints, executes forward passes, and soft-votes (averages softmax probabilities).
2. ensemble_predict falls back cleanly to single-model inference if < 2 valid model paths are provided.
3. InferenceService reads ENSEMBLE_MODELS and activates ensemble mode when 2+ models are specified.
4. Single-model remains the default when ENSEMBLE_MODELS is blank or < 2 paths.
5. POST /images/{id}/analyze returns 'ensemble: true' when ensemble inference is active.
"""

import io
import json
import pytest
from pathlib import Path
from PIL import Image as PILImage
import numpy as np
import torch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import settings
from app.models.case import Case
from app.models.image import Image
from app.models.patient import Patient
from app.services.ensemble import (
    ensemble_predict,
    resolve_model_path,
    load_model_from_checkpoint,
    DEFAULT_CLASSES,
)
from app.services.inference import inference_service, InferenceService, MANDATORY_DISCLAIMER
from app.services.storage_service import storage_service


def create_test_jpeg(width: int = 224, height: int = 224) -> bytes:
    """Generate in-memory RGB JPEG image bytes."""
    buf = io.BytesIO()
    img = PILImage.new("RGB", (width, height), color=(180, 100, 80))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def create_dummy_checkpoint(path: Path, model_name: str = "resnet18", seed: int = 42) -> Path:
    """Helper to create a reproducible dummy model checkpoint."""
    torch.manual_seed(seed)
    import timm
    model = timm.create_model(model_name, pretrained=False, num_classes=len(DEFAULT_CLASSES))
    torch.save({
        "model_name": model_name,
        "classes": DEFAULT_CLASSES,
        "model_state_dict": model.state_dict(),
        "epoch": 1,
    }, path)
    return path


def test_resolve_model_path(tmp_path: Path):
    """Verifies relative and absolute path resolution."""
    file_path = tmp_path / "model.pt"
    file_path.write_text("dummy", encoding="utf-8")

    assert resolve_model_path(file_path) == file_path.resolve()
    assert resolve_model_path(str(file_path)) == file_path.resolve()
    assert resolve_model_path("non_existent_model_file_123.pt") is None


def test_ensemble_predict_soft_voting(tmp_path: Path):
    """
    Verifies that ensemble_predict:
    - Loads multiple models
    - Computes softmax probabilities
    - Averages probabilities across models
    - Returns standardized dictionary with ensemble=True
    """
    ckpt1 = create_dummy_checkpoint(tmp_path / "m1.pt", model_name="resnet18", seed=1)
    ckpt2 = create_dummy_checkpoint(tmp_path / "m2.pt", model_name="resnet18", seed=2)

    image_tensor = torch.randn(1, 3, 224, 224)

    result = ensemble_predict(
        image_tensor=image_tensor,
        model_paths=[str(ckpt1), str(ckpt2)],
        device=torch.device("cpu"),
    )

    assert result["ensemble"] is True
    assert len(result["models_used"]) == 2
    assert "resnet18" in result["models_used"]
    assert "ensemble(" in result["model_version"]
    assert "predicted_class" in result
    assert "predicted_class_name" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_n"]) == 3
    assert len(result["all_predictions"]) == 7
    assert len(result["probabilities"]) == 7

    # Verify that probabilities sum to approximately 1.0
    total_prob = sum(result["probabilities"].values())
    assert pytest.approx(total_prob, abs=0.01) == 1.0


def test_ensemble_predict_fallback_when_under_two_models(tmp_path: Path):
    """
    Verifies fallback to single-model inference (ensemble=False)
    when fewer than 2 valid paths are passed.
    """
    ckpt1 = create_dummy_checkpoint(tmp_path / "single.pt", model_name="resnet18", seed=10)

    image_tensor = torch.randn(1, 3, 224, 224)

    # 1 valid path provided
    res1 = ensemble_predict(
        image_tensor=image_tensor,
        model_paths=[str(ckpt1)],
        device=torch.device("cpu"),
    )
    assert res1["ensemble"] is False
    assert len(res1["models_used"]) == 1
    assert res1["models_used"][0] == "resnet18"
    assert "predicted_class" in res1
    assert "confidence" in res1


def test_inference_service_is_ensemble_property(monkeypatch, tmp_path: Path):
    """Verifies that is_ensemble and ensemble_model_paths react properly to configuration."""
    service = InferenceService()

    # Default is blank -> is_ensemble is False
    monkeypatch.setattr(settings, "ENSEMBLE_MODELS", "")
    assert service.is_ensemble is False
    assert service.ensemble_model_paths == []

    # 1 path -> is_ensemble is False
    monkeypatch.setattr(settings, "ENSEMBLE_MODELS", "path/to/m1.pt")
    assert service.is_ensemble is False
    assert service.ensemble_model_paths == ["path/to/m1.pt"]

    # 2 paths -> is_ensemble is True
    monkeypatch.setattr(settings, "ENSEMBLE_MODELS", "path/to/m1.pt, path/to/m2.pt")
    assert service.is_ensemble is True
    assert service.ensemble_model_paths == ["path/to/m1.pt", "path/to/m2.pt"]


def test_inference_service_predict_with_ensemble(monkeypatch, tmp_path: Path):
    """
    Verifies that InferenceService.predict executes ensemble inference
    when ENSEMBLE_MODELS contains 2+ paths and returns ensemble: True.
    """
    ckpt1 = create_dummy_checkpoint(tmp_path / "model_a.pt", model_name="resnet18", seed=101)
    ckpt2 = create_dummy_checkpoint(tmp_path / "model_b.pt", model_name="resnet18", seed=102)

    monkeypatch.setattr(settings, "ENSEMBLE_MODELS", f"{ckpt1},{ckpt2}")
    service = InferenceService()
    service.reset_mock()

    dummy_img = PILImage.new("RGB", (224, 224), color=(120, 80, 60))
    res = service.predict(dummy_img, run_segmentation=False)

    assert res["ensemble"] is True
    assert len(res["models_used"]) == 2
    assert "ensemble(" in res["model_version"]
    assert res["predicted_class"] in DEFAULT_CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_n"]) == 3
    assert len(res["all_predictions"]) == 7


def test_analyze_image_api_ensemble_response(client: TestClient, db_session: Session, monkeypatch, tmp_path: Path):
    """
    Tests the full API flow POST /images/{id}/analyze when ensemble is active,
    verifying that the response includes ensemble: true.
    """
    ckpt1 = create_dummy_checkpoint(tmp_path / "api_m1.pt", model_name="resnet18", seed=201)
    ckpt2 = create_dummy_checkpoint(tmp_path / "api_m2.pt", model_name="resnet18", seed=202)

    monkeypatch.setattr(settings, "ENSEMBLE_MODELS", f"{ckpt1},{ckpt2}")
    dummy_bytes = create_test_jpeg()
    monkeypatch.setattr(storage_service, "get_file", lambda key: dummy_bytes)

    # Register clinician user
    client.post(
        "/auth/register",
        json={
            "email": "ensemble_clinician@example.com",
            "password": "Password123!",
            "full_name": "Dr. Ensemble",
            "role": "clinician",
        },
    )
    login_res = client.post(
        "/auth/login",
        json={"email": "ensemble_clinician@example.com", "password": "Password123!"},
    )
    token = login_res.json()["access_token"]

    # Setup DB records
    patient = Patient(age=50, gender="female", notes="Ensemble test patient")
    db_session.add(patient)
    db_session.commit()

    case = Case(patient_id=patient.id, created_by=1, body_location="Arm", status="submitted")
    db_session.add(case)
    db_session.commit()

    image = Image(case_id=case.id, storage_key="cases/1/arm.jpg", quality_label="good")
    db_session.add(image)
    db_session.commit()

    # Reset mock on singleton to use the real ensemble code
    inference_service.reset_mock()

    res = client.post(
        f"/images/{image.id}/analyze",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert res.status_code == 200
    data = res.json()
    assert data["ensemble"] is True
    assert "ensemble(" in data["model_version"]
    assert "predicted_class" in data
    assert "confidence" in data
    assert data["disclaimer"] == MANDATORY_DISCLAIMER

