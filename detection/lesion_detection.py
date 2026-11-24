import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_SIZE = 512

# Ignore extreme outer image area
ROI_MARGIN = 0.06

# Plausible lesion size
MIN_AREA_RATIO = 0.005
MAX_AREA_RATIO = 0.45

# Morphological kernel
MORPH_KERNEL_SIZE = 11

# Bounding-box padding
BOX_PADDING = 8


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    """
    Load image using OpenCV.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image


# ============================================================
# IMAGE RESIZING
# ============================================================

def resize_image(image, size=OUTPUT_SIZE):
    """
    Resize image to square size.
    """
    return cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_channel(channel):
    """
    Normalize an image channel to 0-255.
    """
    channel = channel.astype(np.float32)

    min_val = np.min(channel)
    max_val = np.max(channel)

    if max_val - min_val < 1e-6:
        return np.zeros_like(channel, dtype=np.uint8)

    normalized = (
        (channel - min_val)
        / (max_val - min_val)
        * 255.0
    )

    return normalized.astype(np.uint8)


# ============================================================
# CREATE CENTER ROI
# ============================================================

def create_center_roi(image):
    """
    Create a mask that ignores the extreme outer border.
    """

    h, w = image.shape[:2]

    margin_x = int(w * ROI_MARGIN)
    margin_y = int(h * ROI_MARGIN)

    roi_mask = np.zeros((h, w), dtype=np.uint8)

    cv2.rectangle(
        roi_mask,
        (margin_x, margin_y),
        (w - margin_x - 1, h - margin_y - 1),
        255,
        -1
    )

    return roi_mask


# ============================================================
# LESION CANDIDATE MASK
# ============================================================

def create_lesion_candidate_mask(image):
    """
    Create a lesion candidate mask using:
        1. Darkness
        2. Color/chroma difference
        3. Local contrast

    A center ROI is used to reduce false detections
    caused by image borders and background artifacts.
    """

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # Slight smoothing
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    # --------------------------------------------------------
    # Color spaces
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2GRAY
    )

    lab = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2LAB
    )

    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # 1. DARKNESS FEATURE
    # --------------------------------------------------------

    darkness = 255 - gray
    darkness = normalize_channel(darkness)

    # Otsu threshold
    _, dark_mask = cv2.threshold(
        darkness,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # 2. COLOR / CHROMA FEATURE
    # --------------------------------------------------------

    lab_a = lab[:, :, 1].astype(np.float32)
    lab_b = lab[:, :, 2].astype(np.float32)

    # Distance from neutral Lab color
    chroma = np.sqrt(
        (lab_a - 128.0) ** 2 +
        (lab_b - 128.0) ** 2
    )

    chroma = normalize_channel(chroma)

    _, color_mask = cv2.threshold(
        chroma,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # 3. SATURATION FEATURE
    # --------------------------------------------------------

    saturation = hsv[:, :, 1]

    saturation = normalize_channel(saturation)

    _, sat_mask = cv2.threshold(
        saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # 4. LOCAL CONTRAST FEATURE
    # --------------------------------------------------------

    local_background = cv2.GaussianBlur(
        gray,
        (51, 51),
        0
    )

    local_difference = cv2.absdiff(
        gray,
        local_background
    )

    local_difference = normalize_channel(
        local_difference
    )

    _, contrast_mask = cv2.threshold(
        local_difference,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # COMBINE FEATURES
    # --------------------------------------------------------

    # A lesion is more likely to have:
    # - darkness OR
    # - color difference
    # - supported by local contrast

    combined = np.zeros_like(gray)

    dark_binary = dark_mask > 0
    color_binary = color_mask > 0
    sat_binary = sat_mask > 0
    contrast_binary = contrast_mask > 0

    # Voting approach
    votes = (
        dark_binary.astype(np.uint8)
        + color_binary.astype(np.uint8)
        + sat_binary.astype(np.uint8)
        + contrast_binary.astype(np.uint8)
    )

    combined[votes >= 2] = 255

    # --------------------------------------------------------
    # Apply center ROI
    # --------------------------------------------------------

    roi_mask = create_center_roi(image)

    combined = cv2.bitwise_and(
        combined,
        roi_mask
    )

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE
        )
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # Additional smoothing
    # --------------------------------------------------------

    combined = cv2.GaussianBlur(
        combined,
        (5, 5),
        0
    )

    _, combined = cv2.threshold(
        combined,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return combined


# ============================================================
# CONTOUR SELECTION
# ============================================================

def find_best_lesion_contour(mask):
    """
    Find the most plausible lesion contour.

    The contour is scored using:
        - area
        - compactness
        - center proximity
        - non-border contact
    """

    h, w = mask.shape[:2]

    image_area = float(h * w)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    center_x = w / 2.0
    center_y = h / 2.0

    best_contour = None
    best_score = -1.0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area <= 0:
            continue

        area_ratio = area / image_area

        # Reject tiny noise
        if area_ratio < MIN_AREA_RATIO:
            continue

        # Reject giant regions
        if area_ratio > MAX_AREA_RATIO:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        # ----------------------------------------------------
        # Reject contours touching the ROI/image boundary
        # ----------------------------------------------------

        if x <= 1 or y <= 1:
            continue

        if x + bw >= w - 1:
            continue

        if y + bh >= h - 1:
            continue

        # ----------------------------------------------------
        # Center proximity
        # ----------------------------------------------------

        contour_moments = cv2.moments(contour)

        if contour_moments["m00"] != 0:
            cx = (
                contour_moments["m10"]
                / contour_moments["m00"]
            )

            cy = (
                contour_moments["m01"]
                / contour_moments["m00"]
            )
        else:
            cx = x + bw / 2.0
            cy = y + bh / 2.0

        distance = np.sqrt(
            ((cx - center_x) / w) ** 2 +
            ((cy - center_y) / h) ** 2
        )

        center_score = max(
            0.0,
            1.0 - distance / 0.71
        )

        # ----------------------------------------------------
        # Compactness
        # ----------------------------------------------------

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter > 0:
            compactness = (
                4.0 * np.pi * area
                / (perimeter * perimeter)
            )
        else:
            compactness = 0.0

        compactness = min(
            max(compactness, 0.0),
            1.0
        )

        # ----------------------------------------------------
        # Prefer moderate-size regions
        # ----------------------------------------------------

        size_score = min(
            area_ratio / 0.12,
            1.0
        )

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        score = (
            0.45 * center_score
            + 0.30 * size_score
            + 0.25 * compactness
        )

        if score > best_score:
            best_score = score
            best_contour = contour

    return best_contour


# ============================================================
# BOUNDING BOX
# ============================================================

def get_bounding_box(contour, image_shape):
    """
    Generate padded bounding box.

    Returns both short and descriptive key names
    so the detector remains compatible with the
    existing testing/visualization scripts.
    """

    h, w = image_shape[:2]

    x, y, bw, bh = cv2.boundingRect(contour)

    x1 = max(0, x - BOX_PADDING)
    y1 = max(0, y - BOX_PADDING)

    x2 = min(w - 1, x + bw + BOX_PADDING)
    y2 = min(h - 1, y + bh + BOX_PADDING)

    box_width = int(x2 - x1)
    box_height = int(y2 - y1)

    return {
        "x": int(x1),
        "y": int(y1),

        # Short names
        "w": box_width,
        "h": box_height,

        # Descriptive names expected by existing scripts
        "width": box_width,
        "height": box_height
    }


# ============================================================
# LESION MASK
# ============================================================

def create_lesion_mask(contour, image_shape):
    """
    Create binary mask from selected contour.
    """

    h, w = image_shape[:2]

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    cv2.drawContours(
        mask,
        [contour],
        -1,
        255,
        thickness=-1
    )

    return mask


# ============================================================
# VISUALIZATION
# ============================================================

def create_detection_visualization(
    image,
    contour,
    bounding_box
):
    """
    Draw lesion contour and bounding box.
    """

    visualization = image.copy()

    # Draw contour
    cv2.drawContours(
        visualization,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    # Read bounding box
    x = int(bounding_box["x"])
    y = int(bounding_box["y"])
    width = int(bounding_box["width"])
    height = int(bounding_box["height"])

    # Draw bounding rectangle
    cv2.rectangle(
        visualization,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        3
    )

    # Label
    cv2.putText(
        visualization,
        "Lesion Candidate",
        (x, max(y - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )

    return visualization

# ============================================================
# MAIN DETECTION FUNCTION
# ============================================================

def detect_lesion(image_path):
    """
    Complete lesion localization pipeline.

    Returns:
        image_path
        original dimensions
        detected
        bounding_box
        mask
        visualization
    """

    image_path = Path(image_path)

    original_image = load_image(
        image_path
    )

    original_height, original_width = (
        original_image.shape[:2]
    )

    image = resize_image(
        original_image,
        OUTPUT_SIZE
    )

    candidate_mask = (
        create_lesion_candidate_mask(image)
    )

    contour = find_best_lesion_contour(
        candidate_mask
    )

    if contour is None:

        return {
            "image_path": str(image_path),
            "original_width": original_width,
            "original_height": original_height,
            "detected": False,
            "bounding_box": None,
            "mask": candidate_mask,
            "visualization": image
        }

    bounding_box = get_bounding_box(
        contour,
        image.shape
    )

    lesion_mask = create_lesion_mask(
        contour,
        image.shape
    )

    visualization = (
        create_detection_visualization(
            image,
            contour,
            bounding_box
        )
    )

    return {
        "image_path": str(image_path),
        "original_width": original_width,
        "original_height": original_height,
        "detected": True,
        "bounding_box": bounding_box,
        "mask": lesion_mask,
        "visualization": visualization
    }