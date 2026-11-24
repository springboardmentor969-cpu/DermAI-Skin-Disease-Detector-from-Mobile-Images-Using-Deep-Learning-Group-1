from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MIN_WIDTH = 100
MIN_HEIGHT = 100

# These are initial engineering thresholds.
# They can be calibrated after running the assessment
# over the PAD-UFES-20 dataset.

MIN_BLUR_SCORE = 50.0

MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 220.0

MIN_CONTRAST = 20.0


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    """
    Load an image using OpenCV.

    Returns:
        image: BGR image
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    return image


# ============================================================
# BLUR / SHARPNESS
# ============================================================

def calculate_blur_score(image):
    """
    Calculate image sharpness using
    variance of the Laplacian.

    Higher value generally indicates
    stronger image detail/sharpness.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return float(laplacian.var())


# ============================================================
# BRIGHTNESS
# ============================================================

def calculate_brightness(image):
    """
    Calculate mean grayscale intensity.

    Range:
        approximately 0 - 255
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(gray.mean())


# ============================================================
# CONTRAST
# ============================================================

def calculate_contrast(image):
    """
    Calculate grayscale standard deviation
    as a simple contrast measure.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(gray.std())


# ============================================================
# IMAGE DIMENSIONS
# ============================================================

def check_dimensions(image):
    """
    Check whether image dimensions are
    large enough for processing.
    """

    height, width = image.shape[:2]

    valid = (
        width >= MIN_WIDTH
        and height >= MIN_HEIGHT
    )

    return {
        "width": width,
        "height": height,
        "valid": valid,
    }


# ============================================================
# COMPLETE IMAGE QUALITY ASSESSMENT
# ============================================================

def assess_image_quality(image_path):
    """
    Perform complete image quality assessment.

    Returns a dictionary containing:

        - readability
        - dimensions
        - blur score
        - brightness
        - contrast
        - individual quality checks
        - overall quality status
    """

    result = {
        "image_path": str(image_path),
        "readable": False,
        "width": 0,
        "height": 0,
        "blur_score": 0.0,
        "brightness": 0.0,
        "contrast": 0.0,
        "blur_ok": False,
        "brightness_ok": False,
        "contrast_ok": False,
        "dimensions_ok": False,
        "quality_status": "INVALID",
    }

    try:
        image = load_image(image_path)

        result["readable"] = True

        # Dimensions
        dimension_result = check_dimensions(image)

        result["width"] = dimension_result["width"]
        result["height"] = dimension_result["height"]
        result["dimensions_ok"] = dimension_result["valid"]

        # Blur / sharpness
        blur_score = calculate_blur_score(image)

        result["blur_score"] = round(
            blur_score,
            4
        )

        result["blur_ok"] = (
            blur_score >= MIN_BLUR_SCORE
        )

        # Brightness
        brightness = calculate_brightness(image)

        result["brightness"] = round(
            brightness,
            4
        )

        result["brightness_ok"] = (
            MIN_BRIGHTNESS
            <= brightness
            <= MAX_BRIGHTNESS
        )

        # Contrast
        contrast = calculate_contrast(image)

        result["contrast"] = round(
            contrast,
            4
        )

        result["contrast_ok"] = (
            contrast >= MIN_CONTRAST
        )

        # Overall status
        checks = [
            result["dimensions_ok"],
            result["blur_ok"],
            result["brightness_ok"],
            result["contrast_ok"],
        ]

        passed = sum(checks)

        if passed == 4:
            result["quality_status"] = "GOOD"

        elif passed >= 2:
            result["quality_status"] = "WARNING"

        else:
            result["quality_status"] = "POOR"

    except Exception as exc:
        result["quality_status"] = "INVALID"
        result["error"] = str(exc)

    return result