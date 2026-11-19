"""
Skin Lesion Image Quality Assessment Module.

Computes blur score (Laplacian variance), brightness, contrast, and resolution metrics,
classifying the image into 'good', 'acceptable', or 'poor' based on configurable thresholds.
"""

import io
from pathlib import Path
from typing import Dict, Any, Union, List
import cv2
import numpy as np
from PIL import Image as PILImage

# ------------------------------------------------------------------------------
# Configurable Quality Thresholds
# Empirically calibrated from N=1,000 random sample of HAM10000 dermoscopy images
# (via ml/preprocessing/calibrate_thresholds.py).
# Calibrated targeting ~92.7% 'good', ~6.0% 'acceptable', and ~1.3% 'poor' (extreme outliers).
# ------------------------------------------------------------------------------
QUALITY_THRESHOLDS = {
    # Blur (Laplacian variance):
    # HAM10000 median is 49.0, 5th pctl is 21.8, 1st pctl is 15.4.
    # Polarized/fluid dermoscopy naturally produces smooth skin textures without
    # synthetic high-frequency specular edges.
    "blur_poor": 15.0,              # < 15.0 flags extreme blurry outliers (< 1st percentile)
    "blur_acceptable": 20.0,        # 15.0 - 20.0 is borderline / marginally sharp (1st - 5th pctl)

    # Brightness (mean pixel intensity on a 0-255 scale):
    # HAM10000 median is 156.1, 1st pctl is 118.2 (min 102.6), 95th pctl is 188.0, 99th is 199.7.
    "brightness_dark_poor": 100.0,        # < 100.0 is severely underexposed (below HAM10000 min)
    "brightness_dark_acceptable": 115.0,  # 100.0 - 115.0 is somewhat dark (below 1st pctl)
    "brightness_bright_acceptable": 195.0,# 195.0 - 210.0 is somewhat bright (above 95th pctl)
    "brightness_bright_poor": 210.0,     # > 210.0 is washed out / over-exposed (> 99th pctl)

    # Contrast (standard deviation of pixel intensity / RMS contrast):
    # HAM10000 median is 26.1, 1st pctl is 10.0, 5th pctl is 13.3.
    "contrast_poor": 8.0,           # < 8.0 indicates severely low contrast (< 1st pctl)
    "contrast_acceptable": 11.0,    # 8.0 - 11.0 is borderline contrast (1st - 5th pctl)

    # Minimum image dimensions (width or height in pixels)
    "min_dimension_poor": 224,      # Mandatory minimum deep learning CNN input resolution
    "min_dimension_acceptable": 300, # Sub-optimal resolution for dermoscopy screening
}


def _load_image_as_rgb_and_gray(
    image_input: Union[str, Path, bytes, PILImage.Image, np.ndarray]
) -> tuple[PILImage.Image, np.ndarray, np.ndarray]:
    """
    Normalizes diverse image input types into PIL Image (RGB),
    NumPy RGB array, and Grayscale NumPy array.
    """
    if isinstance(image_input, (str, Path)):
        pil_img = PILImage.open(str(image_input))
    elif isinstance(image_input, (bytes, bytearray)):
        pil_img = PILImage.open(io.BytesIO(image_input))
    elif isinstance(image_input, PILImage.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            pil_img = PILImage.fromarray(image_input).convert("RGB")
        elif image_input.shape[2] == 4:
            pil_img = PILImage.fromarray(image_input).convert("RGB")
        else:
            pil_img = PILImage.fromarray(image_input)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Ensure RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    rgb_array = np.array(pil_img)
    gray_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

    return pil_img, rgb_array, gray_array


def assess_image_quality(
    image_input: Union[str, Path, bytes, PILImage.Image, np.ndarray],
    thresholds: Dict[str, Any] = QUALITY_THRESHOLDS,
) -> Dict[str, Any]:
    """
    Evaluate the clinical quality of a skin lesion image.

    Args:
        image_input: Filepath, raw bytes, PIL Image, or NumPy array.
        thresholds: Custom threshold dictionary (defaults to QUALITY_THRESHOLDS).

    Returns:
        dict: {
            "blur_score": float,
            "brightness": float,
            "contrast": float,
            "resolution": {"width": int, "height": int, "megapixels": float},
            "quality_label": "good" | "acceptable" | "poor",
            "issues": list[str]
        }
    """
    pil_img, _, gray = _load_image_as_rgb_and_gray(image_input)
    width, height = pil_img.size

    # 1. Blur score via Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # 2. Brightness (mean pixel intensity)
    brightness = float(np.mean(gray))

    # 3. Contrast (standard deviation of pixel intensity / RMS contrast)
    contrast = float(np.std(gray))

    # Evaluate quality against thresholds
    issues: List[str] = []
    is_poor = False
    is_acceptable = False

    # Check blur
    if blur_score < thresholds["blur_poor"]:
        is_poor = True
        issues.append(f"Image is blurry (blur score {blur_score:.1f} < {thresholds['blur_poor']})")
    elif blur_score < thresholds["blur_acceptable"]:
        is_acceptable = True
        issues.append(f"Borderline sharpness (blur score {blur_score:.1f})")

    # Check brightness
    if brightness < thresholds["brightness_dark_poor"]:
        is_poor = True
        issues.append(f"Image is severely under-exposed (brightness {brightness:.1f})")
    elif brightness > thresholds["brightness_bright_poor"]:
        is_poor = True
        issues.append(f"Image is over-exposed/washed out (brightness {brightness:.1f})")
    elif brightness < thresholds["brightness_dark_acceptable"]:
        is_acceptable = True
        issues.append(f"Image is somewhat dark (brightness {brightness:.1f})")
    elif brightness > thresholds["brightness_bright_acceptable"]:
        is_acceptable = True
        issues.append(f"Image is somewhat bright (brightness {brightness:.1f})")

    # Check contrast
    if contrast < thresholds["contrast_poor"]:
        is_poor = True
        issues.append(f"Contrast is very low ({contrast:.1f} < {thresholds['contrast_poor']})")
    elif contrast < thresholds["contrast_acceptable"]:
        is_acceptable = True
        issues.append(f"Contrast is suboptimal ({contrast:.1f})")

    # Check resolution
    min_dim = min(width, height)
    if min_dim < thresholds["min_dimension_poor"]:
        is_poor = True
        issues.append(f"Resolution is below minimal requirements ({width}x{height} < {thresholds['min_dimension_poor']})")
    elif min_dim < thresholds["min_dimension_acceptable"]:
        is_acceptable = True
        issues.append(f"Low resolution for clinical analysis ({width}x{height})")

    # Overall classification
    if is_poor:
        quality_label = "poor"
    elif is_acceptable:
        quality_label = "acceptable"
    else:
        quality_label = "good"

    return {
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "width": width,
        "height": height,
        "resolution": {
            "width": width,
            "height": height,
            "megapixels": round((width * height) / 1_000_000, 2),
        },
        "quality_label": quality_label,
        "issues": issues,
    }
