"""
Unit and integration tests for DermAI Grad-CAM explainability module.

Tests:
1. ml/explainability/gradcam.py generate_gradcam():
   - Generates heatmap with values normalized to [0, 1]
   - Generates valid PNG overlay bytes
   - Supports CNNs (ResNet, EfficientNet, DenseNet, MobileNet) and ViT / Swin
   - Automatically uses predicted class when target_class is None
2. backend/app/services/inference.py:
   - Calls generate_gradcam after prediction
   - Uploads overlay PNG bytes to MinIO under gradcam/{image_id}.png
   - Returns gradcam_key in the prediction dict
3. backend/app/routers/images.py:
   - GET /images/{image_id}/gradcam endpoint returns image/png
   - Requires JWT authentication
   - Checks image ownership/case authorization
"""

import io
import pytest
from pathlib import Path
from PIL import Image as PILImage
import numpy as np
import torch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import timm

from app.models.case import Case
from app.models.image import Image
from app.models.patient import Patient
from app.models.prediction import Prediction
from app.services.inference import inference_service, InferenceService
from app.services.storage_service import storage_service
from ml.explainability.gradcam import generate_gradcam, detect_target_layers, reshape_transform_vit, reshape_transform_swin


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
    patient = Patient(age=45, gender="male", notes="Grad-CAM test patient")
    db_session.add(patient)
    db_session.commit()
    case = Case(patient_id=patient.id, created_by=created_by, body_location=body_location, status="submitted")
    db_session.add(case)
    db_session.commit()
    return case


def test_generate_gradcam_dummy_random_image_tensor():
    """
    Test generate_gradcam() returns numpy array shape (H,W) with values in [0,1]
    using a dummy random image tensor.
    """
    model = timm.create_model("resnet18", pretrained=False, num_classes=7).eval()
    H, W = 224, 224
    dummy_tensor = torch.randn(1, 3, H, W)
    result = generate_gradcam(model, dummy_tensor)

    assert "heatmap" in result
    heatmap = result["heatmap"]
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (H, W)
    assert 0.0 <= heatmap.min()
    assert heatmap.max() <= 1.0
    assert "overlay_bytes" in result
    assert isinstance(result["overlay_bytes"], bytes)
    assert len(result["overlay_bytes"]) > 0


@pytest.mark.parametrize("model_name", [
    "resnet18",
    "efficientnet_b0",
    "densenet121",
    "mobilenetv3_large_100",
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
])
def test_generate_gradcam_architectures(model_name: str):
    """
    Verifies generate_gradcam auto-detects layers and generates valid
    heatmaps and PNG overlays across all target model families.
    """
    model = timm.create_model(model_name, pretrained=False, num_classes=7).eval()
    inp = torch.randn(1, 3, 224, 224)

    # 1. Test with explicit target class
    res = generate_gradcam(model, inp, target_class=4)
    assert "heatmap" in res
    assert "overlay_bytes" in res
    assert isinstance(res["heatmap"], np.ndarray)
    assert res["heatmap"].shape == (224, 224)
    assert 0.0 <= res["heatmap"].min() <= res["heatmap"].max() <= 1.0

    # Verify PNG bytes header
    assert res["overlay_bytes"][:8] == b"\x89PNG\r\n\x1a\n"
    overlay_img = PILImage.open(io.BytesIO(res["overlay_bytes"]))
    assert overlay_img.size == (224, 224)

    # 2. Test with target_class=None (should use predicted class)
    res_none = generate_gradcam(model, inp, target_class=None)
    assert "heatmap" in res_none
    assert "overlay_bytes" in res_none
    assert res_none["target_class"] in range(7)


def test_inference_service_gradcam_generation(monkeypatch, tmp_path: Path):
    """
    Verifies that InferenceService.predict executes generate_gradcam,
    stores the overlay in MinIO under gradcam/{image_id}.png,
    and returns gradcam_key in the prediction result dictionary.
    """
    uploaded_files = {}

    def mock_upload_file(file_bytes, storage_key, content_type, bucket_name=None):
        uploaded_files[storage_key] = file_bytes
        return storage_key

    monkeypatch.setattr(storage_service, "upload_file", mock_upload_file)

    # Create dummy model checkpoint for resnet18
    model_file = tmp_path / "model.pt"
    m = timm.create_model("resnet18", pretrained=False, num_classes=7)
    torch.save({
        "model_name": "resnet18",
        "model_state_dict": m.state_dict(),
    }, model_file)

    service = InferenceService()
    service.reset_mock()
    service.load_model(model_file)

    dummy_img = PILImage.new("RGB", (224, 224), color=(120, 80, 60))
    res = service.predict(dummy_img, run_segmentation=False, image_id=42)

    assert "gradcam_key" in res
    assert res["gradcam_key"] == "gradcam/42.png"
    assert "gradcam/42.png" in uploaded_files
    assert uploaded_files["gradcam/42.png"][:8] == b"\x89PNG\r\n\x1a\n"


def test_get_image_gradcam_endpoint(client: TestClient, db_session: Session, monkeypatch):
    """
    Verifies GET /images/{image_id}/gradcam:
    - Returns 200 with media_type image/png
    - Requires JWT authentication
    - Verifies image access permissions
    """
    clinician_token = get_auth_token(client, "gradcam_doc@example.com", "clinician")
    other_patient_token = get_auth_token(client, "other_patient@example.com", "patient")

    case = create_case_with_patient(db_session, created_by=1, body_location="Arm")
    image = Image(case_id=case.id, storage_key="cases/1/arm.jpg", quality_label="good")
    db_session.add(image)
    db_session.commit()

    # Generate dummy PNG bytes
    dummy_png = io.BytesIO()
    PILImage.new("RGB", (224, 224), color=(255, 0, 0)).save(dummy_png, format="PNG")
    dummy_bytes = dummy_png.getvalue()

    monkeypatch.setattr(storage_service, "get_file", lambda key: dummy_bytes)

    # 1. Unauthenticated request -> 401
    unauth_res = client.get(f"/images/{image.id}/gradcam")
    assert unauth_res.status_code == 401

    # 2. Clinician request -> 200 with image/png for default overlay
    doc_res = client.get(
        f"/images/{image.id}/gradcam",
        headers={"Authorization": f"Bearer {clinician_token}"},
    )
    assert doc_res.status_code == 200
    assert doc_res.headers["content-type"] == "image/png"
    assert doc_res.content[:8] == b"\x89PNG\r\n\x1a\n"

    # Test ?type=heatmap and ?type=overlay query params
    heat_res = client.get(
        f"/images/{image.id}/gradcam?type=heatmap",
        headers={"Authorization": f"Bearer {clinician_token}"},
    )
    assert heat_res.status_code == 200
    assert heat_res.headers["content-type"] == "image/png"

    overlay_res = client.get(
        f"/images/{image.id}/gradcam?type=overlay",
        headers={"Authorization": f"Bearer {clinician_token}"},
    )
    assert overlay_res.status_code == 200
    assert overlay_res.headers["content-type"] == "image/png"

    # 3. Unauthorized patient request -> 403
    forbidden_res = client.get(
        f"/images/{image.id}/gradcam",
        headers={"Authorization": f"Bearer {other_patient_token}"},
    )
    assert forbidden_res.status_code == 403

