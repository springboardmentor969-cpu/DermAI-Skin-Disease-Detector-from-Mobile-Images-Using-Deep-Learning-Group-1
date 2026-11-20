from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT
from app.main import app
from app.services.classification_service import (
    get_model_for_explainability,
    get_models_comparison_metrics,
    predict_skin_disease,
)
from app.services.explainability_service import generate_gradcam_explanation
from app.services.risk_service import assess_skin_disease_risk


@pytest.fixture
def sample_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(180, 120, 90))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_risk_assessment_categories():
    # 1. Melanoma -> High Risk
    risk_mel = assess_skin_disease_risk("mel", confidence=0.85)
    assert risk_mel["risk_level"] == "High Risk"
    assert risk_mel["urgency"] == "Urgent"
    assert "Melanoma" in risk_mel["clinical_name"]
    assert "disclaimer" in risk_mel

    # 2. Basal Cell Carcinoma -> High Risk
    risk_bcc = assess_skin_disease_risk("bcc", confidence=0.90)
    assert risk_bcc["risk_level"] == "High Risk"

    # 3. Actinic Keratosis -> Moderate Risk
    risk_akiec = assess_skin_disease_risk("akiec", confidence=0.75)
    assert risk_akiec["risk_level"] == "Moderate Risk"

    # 4. Nevus -> Low Risk
    risk_nv = assess_skin_disease_risk("nv", confidence=0.88)
    assert risk_nv["risk_level"] == "Low Risk"

    # 5. Low confidence triggers Clinical Review
    risk_low_conf = assess_skin_disease_risk("nv", confidence=0.35)
    assert risk_low_conf["risk_level"] == "Requires Clinical Review"

    # 6. Poor quality image triggers Clinical Review
    risk_poor_qual = assess_skin_disease_risk("nv", confidence=0.95, is_poor_quality=True)
    assert risk_poor_qual["risk_level"] == "Requires Clinical Review"


def test_gradcam_explainability(sample_image_bytes):
    model, arch = get_model_for_explainability("mobilenet_v2")
    explanation = generate_gradcam_explanation(model, arch, sample_image_bytes, target_class_index=5)

    assert explanation["status"] == "available"
    assert "Grad-CAM" in explanation["method"]
    assert explanation["heatmap_data_url"].startswith("data:image/png;base64,")
    assert explanation["overlay_data_url"].startswith("data:image/png;base64,")
    assert len(explanation["peak_attention_coords"]) == 2
    assert 0 <= explanation["attention_coverage_percent"] <= 100
    assert "attention" in explanation["interpretation"].lower()


def test_models_comparison_metrics_structure():
    metrics = get_models_comparison_metrics()
    assert "models" in metrics
    assert "classes" in metrics
    assert len(metrics["classes"]) == 7

    model_ids = [m["id"] for m in metrics["models"]]
    assert "mobilenet_v2" in model_ids
    assert "resnet50" in model_ids
    assert "vit_b_16" in model_ids
    assert "ensemble" in model_ids

    for m in metrics["models"]:
        assert 0.0 < m["validation_accuracy"] <= 1.0
        assert 0.0 < m["validation_macro_f1"] <= 1.0
        assert 0.0 < m["test_accuracy"] <= 1.0
        assert 0.0 < m["auc_roc"] <= 1.0
        assert m["parameters_m"] > 0


def test_multimodel_prediction_and_ensemble(sample_image_bytes):
    # Test baseline CNN
    res_mobilenet = predict_skin_disease(sample_image_bytes, architecture="mobilenet_v2")
    assert res_mobilenet["status"] == "available"
    assert res_mobilenet["predicted_class"] in [
        "akiec",
        "bcc",
        "bkl",
        "df",
        "mel",
        "nv",
        "vasc",
    ]
    assert 0.0 <= res_mobilenet["confidence"] <= 1.0
    assert "uncertainty_entropy" in res_mobilenet
    assert len(res_mobilenet["top_predictions"]) == 7

    # Test Ensemble
    res_ensemble = predict_skin_disease(sample_image_bytes, architecture="ensemble")
    assert res_ensemble["status"] == "available"
    assert res_ensemble["architecture"] == "ensemble"
    assert 0.0 <= res_ensemble["confidence"] <= 1.0


def test_models_endpoint():
    client = TestClient(app)
    response = client.get("/api/v1/analysis/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert any(m["id"] == "resnet50" for m in data["models"])
    assert any(m["id"] == "vit_b_16" for m in data["models"])
    assert any(m["id"] == "ensemble" for m in data["models"])

