"""
Machine Learning & Risk Scorer Pipeline Tests for DermAI.

Tests:
1. predict() returns dict with keys: predicted_class, confidence, probabilities.
2. probabilities has all 7 keys: akiec, bcc, bkl, df, mel, nv, vasc.
3. probabilities values sum to 1.0 (within 0.001 tolerance).
4. Different input images return different predicted_class values.
5. Risk scorer returns HIGH for mel class regardless of confidence.
6. Risk scorer returns LOW for nv with confidence >= 0.5.
"""

import io
from pathlib import Path
import pytest
import numpy as np
from PIL import Image as PILImage

from app.services.inference import inference_service
from app.services.risk_scorer import compute_risk

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def make_test_image(color: tuple, pattern_id: int = 0) -> bytes:
    """Creates a distinct 250x250 test image for ML inference."""
    img = PILImage.new("RGB", (250, 250), color=color)
    # Add distinct patterned pixels so image hashes differ significantly
    pixels = img.load()
    for x in range(pattern_id * 20, (pattern_id + 1) * 20):
        for y in range(50):
            pixels[x, y] = (0, 0, 0)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Returns sample lesion JPEG bytes from ml/data or generated test image."""
    p = PROJECT_ROOT / "ml" / "data" / "isic2018_task1" / "images" / "ISIC_0024386.jpg"
    if p.is_file():
        return p.read_bytes()
    return make_test_image((160, 90, 80), pattern_id=1)


def test_predict_returns_required_keys(sample_image_bytes):
    """predict() must return dict with keys: predicted_class, confidence, probabilities."""
    res = inference_service.predict(sample_image_bytes)
    assert isinstance(res, dict)
    for required_key in ["predicted_class", "confidence", "probabilities"]:
        assert required_key in res, f"Missing required key '{required_key}' in predict() result"
    assert isinstance(res["predicted_class"], str)
    assert isinstance(res["confidence"], float)
    assert isinstance(res["probabilities"], dict)


def test_probabilities_has_all_seven_classes_and_sums_to_one(sample_image_bytes):
    """probabilities dict must contain all 7 classes and sum to 1.0 within 0.001 tolerance."""
    res = inference_service.predict(sample_image_bytes)
    probs = res["probabilities"]

    expected_classes = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    assert set(probs.keys()) == expected_classes, f"Mismatch in expected 7 classes: {probs.keys()}"

    for cls_name, prob_val in probs.items():
        assert 0.0 <= prob_val <= 1.0, f"Probability for {cls_name} out of bounds: {prob_val}"

    total_prob = sum(probs.values())
    assert pytest.approx(total_prob, abs=0.001) == 1.0, f"Probabilities sum {total_prob} is not 1.0"


def test_different_input_images_return_different_classes():
    """Different input images must return different predicted_class values."""
    # Test with 3 distinct images
    img_a = make_test_image((210, 140, 100), pattern_id=1)
    img_b = make_test_image((70, 40, 30), pattern_id=5)
    img_c = make_test_image((140, 180, 160), pattern_id=10)

    # Check against ISIC samples if available
    isic_a = PROJECT_ROOT / "ml" / "data" / "isic2018_task1" / "images" / "ISIC_0024386.jpg"
    isic_b = PROJECT_ROOT / "ml" / "data" / "isic2018_task1" / "images" / "ISIC_0024838.jpg"
    isic_c = PROJECT_ROOT / "ml" / "data" / "isic2018_task1" / "images" / "ISIC_0025071.jpg"

    if isic_a.is_file() and isic_b.is_file() and isic_c.is_file():
        img_a = isic_a.read_bytes()
        img_b = isic_b.read_bytes()
        img_c = isic_c.read_bytes()

    res_a = inference_service.predict(img_a)
    res_b = inference_service.predict(img_b)
    res_c = inference_service.predict(img_c)

    classes = [res_a["predicted_class"], res_b["predicted_class"], res_c["predicted_class"]]
    unique_classes = set(classes)

    assert len(unique_classes) > 1, f"Expected varied predictions across distinct images, got: {classes}"


def test_risk_scorer_mel_is_high_at_any_confidence():
    """Risk scorer returns HIGH for mel (Melanoma) class at any confidence level."""
    for conf in [0.15, 0.35, 0.50, 0.88, 0.99]:
        risk = compute_risk("mel", conf)
        assert risk["risk_level"] == "HIGH", f"Melanoma at confidence {conf} expected HIGH, got {risk['risk_level']}"
        assert risk["risk_color"] == "red"
        assert risk["urgency_days"] == 1


def test_risk_scorer_nv_is_low_at_confidence_above_half():
    """Risk scorer returns LOW for nv (Nevus) with confidence >= 0.5."""
    for conf in [0.50, 0.65, 0.80, 0.95]:
        risk = compute_risk("nv", conf)
        assert risk["risk_level"] == "LOW", f"Nevus at confidence {conf} expected LOW, got {risk['risk_level']}"
        assert risk["risk_color"] == "green"
        assert risk["urgency_days"] == 90
