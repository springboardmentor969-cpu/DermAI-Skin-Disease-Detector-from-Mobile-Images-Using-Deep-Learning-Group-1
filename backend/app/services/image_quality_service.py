from __future__ import annotations

import cv2
import numpy as np


def assess_image_quality(image_bytes: bytes) -> dict[str, object]:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image quality assessment could not decode the image.")

    height, width = image.shape[:2]
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    brightness = float(grayscale.mean())
    contrast = float(grayscale.std())
    underexposed_fraction = float(np.mean(grayscale <= 20))
    overexposed_fraction = float(np.mean(grayscale >= 235))
    noise_residual = cv2.absdiff(
        grayscale, cv2.GaussianBlur(grayscale, (0, 0), sigmaX=1.0)
    )
    noise_level = float(noise_residual.std())

    sharpness_score = float(np.clip((sharpness - 20.0) / 280.0 * 100.0, 0, 100))
    exposure_penalty = (
        abs(brightness - 127.5) / 127.5 * 55.0
        + max(underexposed_fraction - 0.02, 0.0) * 250.0
        + max(overexposed_fraction - 0.02, 0.0) * 250.0
    )
    exposure_score = float(np.clip(100.0 - exposure_penalty, 0, 100))
    contrast_score = float(np.clip((contrast - 10.0) / 45.0 * 100.0, 0, 100))
    resolution_score = float(np.clip(min(width, height) * 100.0 / 512.0, 0, 100))
    noise_score = float(np.clip(100.0 - max(noise_level - 3.0, 0.0) * 8.0, 0, 100))
    weights = {
        "sharpness": 0.35,
        "exposure": 0.25,
        "contrast": 0.20,
        "resolution": 0.15,
        "artifacts": 0.05,
    }
    score = round(
        weights["sharpness"] * sharpness_score
        + weights["exposure"] * exposure_score
        + weights["contrast"] * contrast_score
        + weights["resolution"] * resolution_score
        + weights["artifacts"] * noise_score,
        1,
    )
    if score >= 80:
        category = "good"
    elif score >= 60:
        category = "fair"
    else:
        category = "poor"
    feedback: list[str] = []
    if sharpness_score < 50:
        feedback.append("Image may be blurred; hold the camera steady and recapture.")
    if underexposed_fraction > 0.02 or brightness < 35:
        feedback.append("Image appears underexposed; use brighter, even lighting.")
    if overexposed_fraction > 0.02 or brightness > 220:
        feedback.append("Image appears overexposed; reduce glare and avoid direct flash.")
    if contrast_score < 40:
        feedback.append("Image has low contrast; improve lighting and lesion visibility.")
    if noise_score < 50:
        feedback.append("Image contains substantial noise or compression-like artifacts.")
    if min(width, height) < 256:
        feedback.append("Image resolution is low; recapture at a higher resolution.")
    recapture_recommended = category == "poor" or bool(feedback)

    return {
        "score": score,
        "category": category,
        "method": "heuristic",
        "weights": weights,
        "metrics": {
            "laplacian_variance": round(sharpness, 2),
            "mean_brightness": round(brightness, 2),
            "contrast_stddev": round(contrast, 2),
            "underexposed_fraction": round(underexposed_fraction, 4),
            "overexposed_fraction": round(overexposed_fraction, 4),
            "noise_residual_stddev": round(noise_level, 2),
            "width": width,
            "height": height,
        },
        "components": {
            "sharpness": round(sharpness_score, 1),
            "exposure": round(exposure_score, 1),
            "contrast": round(contrast_score, 1),
            "resolution": round(resolution_score, 1),
            "artifacts": round(noise_score, 1),
        },
        "feedback": feedback or ["Image quality is acceptable for technical review."],
        "recapture_recommended": recapture_recommended,
        "disclaimer": "A technical image-quality estimate; not a diagnostic assessment.",
    }
