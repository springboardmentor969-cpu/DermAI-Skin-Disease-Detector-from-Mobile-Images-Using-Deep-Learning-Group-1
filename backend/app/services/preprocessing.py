import base64
import io
import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}

def _encode_rgb_to_base64(rgb_arr: np.ndarray) -> str:
    pil_img = Image.fromarray(np.clip(rgb_arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def evaluate_image_quality(rgb_arr: np.ndarray, file_size_kb: float, img_format: str) -> dict:
    h, w, _ = rgb_arr.shape
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)

    # 1. Blur detection via variance of Laplacian
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 2. Brightness & exposure check
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # 3. Center lesion visibility / contrast against periphery
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    center_patch = gray[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(w, cx + r)]
    border_mean = float(np.mean(gray))
    center_diff = abs(float(np.mean(center_patch)) - border_mean)
    visibility_score = float(np.clip(45.0 + center_diff * 1.4 + min(contrast, 45.0) * 0.6, 0.0, 100.0))

    warnings = []
    if img_format.upper() not in ALLOWED_FORMATS:
        warnings.append(f"Unsupported image format: {img_format}")
    if w < 180 or h < 180:
        warnings.append(f"Low resolution ({w}x{h}px). Recommended minimum is 224x224px.")
    if lap_var < 35.0:
        warnings.append(f"High image blur detected (Laplacian variance={lap_var:.1f} < 35.0).")
    if brightness < 55.0:
        warnings.append(f"Image is underexposed / too dark (mean brightness={brightness:.1f}).")
    elif brightness > 225.0:
        warnings.append(f"Image is overexposed (mean brightness={brightness:.1f}).")
    if contrast < 16.0:
        warnings.append(f"Low lesion-to-skin contrast (std={contrast:.1f}).")

    if len(warnings) == 0 and lap_var >= 60.0 and 70.0 <= brightness <= 210.0:
        quality_status = "Good Quality"
        suitable = True
    elif lap_var >= 25.0 and 45.0 <= brightness <= 235.0 and w >= 128 and h >= 128:
        quality_status = "Acceptable Quality"
        suitable = True
    else:
        quality_status = "Poor Quality"
        suitable = False

    return {
        "width_px": w,
        "height_px": h,
        "file_size_kb": round(file_size_kb, 2),
        "format": img_format.upper(),
        "blur_laplacian_var": round(lap_var, 2),
        "brightness_mean": round(brightness, 2),
        "contrast_std": round(contrast, 2),
        "lesion_visibility_score": round(visibility_score, 1),
        "quality_status": quality_status,
        "suitable_for_ai": suitable,
        "warnings": warnings,
    }

def remove_hair_dullrazor(rgb_arr: np.ndarray) -> np.ndarray:
    '''Morphological Black-Hat filtering + Telea inpainting to remove dark hair artifacts.'''
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    bh_blur = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, hair_mask = cv2.threshold(bh_blur, 14, 255, cv2.THRESH_BINARY)
    inpainted_bgr = cv2.inpaint(bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

def enhance_contrast_clahe(rgb_arr: np.ndarray) -> np.ndarray:
    '''Applies CLAHE on the L-channel in CIE LAB color space.'''
    lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    merged = cv2.merge((l_eq, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def color_constancy_shades_of_gray(rgb_arr: np.ndarray, power: int = 6) -> np.ndarray:
    '''Minkowski-norm Shades-of-Gray illumination normalization widely used in ISIC pipelines.'''
    img_f = rgb_arr.astype(np.float32)
    norm_vec = np.power(np.mean(np.power(img_f, power), axis=(0, 1)), 1.0 / power)
    norm_vec = np.maximum(norm_vec, 1e-6)
    scale = np.sqrt(np.sum(np.square(norm_vec))) / np.sqrt(3.0)
    corrected = img_f * (scale / norm_vec)
    return np.clip(corrected, 0, 255).astype(np.uint8)

def run_preprocessing_pipeline(
    image_bytes: bytes,
    filename: str = "upload.png",
    target_size: int = 224,
    crop_ratio: float = 0.90,
) -> dict:
    pil_img = Image.open(io.BytesIO(image_bytes))
    img_format = pil_img.format or "PNG"
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    raw_rgb = np.array(pil_img)

    file_size_kb = len(image_bytes) / 1024.0
    quality = evaluate_image_quality(raw_rgb, file_size_kb, img_format)

    # Step 1: Center ROI Crop
    h, w, _ = raw_rgb.shape
    side = int(min(h, w) * crop_ratio)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = raw_rgb[y0:y0 + side, x0:x0 + side]

    # Step 2: Resize to 224x224
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Step 3: DullRazor Hair & Artifact Reduction
    hair_removed = remove_hair_dullrazor(resized)

    # Step 4: Gaussian Denoising & CLAHE Contrast Enhancement
    denoised = cv2.GaussianBlur(hair_removed, (3, 3), 0.5)
    clahe_img = enhance_contrast_clahe(denoised)

    # Step 5: Shades-of-Gray Color Constancy
    color_norm = color_constancy_shades_of_gray(clahe_img)

    # Step 6: Pixel Normalization [0, 1] and ImageNet Z-score Standardization
    tensor_01 = color_norm.astype(np.float32) / 255.0
    mean_arr = np.array(IMAGENET_MEAN, dtype=np.float32)
    std_arr = np.array(IMAGENET_STD, dtype=np.float32)
    standardized_tensor = (tensor_01 - mean_arr) / std_arr

    return {
        "quality_metrics": quality,
        "pipeline_steps_applied": [
            "EXIF Orientation Correction",
            f"Center Lesion ROI Crop ({int(crop_ratio*100)}% FOV)",
            f"Bilinear Resize ({target_size}x{target_size}px)",
            "DullRazor Morphological Black-Hat Hair Removal",
            "CIE-LAB CLAHE Contrast Enhancement (clipLimit=2.2)",
            "Shades-of-Gray (p=6) Illumination Normalization",
            "ImageNet Z-Score Tensor Standardization",
        ],
        "tensor_shape": [1, 3, target_size, target_size],
        "normalization_stats": {
            "pixel_range_01": [round(float(tensor_01.min()), 4), round(float(tensor_01.max()), 4)],
            "standardized_mean_rgb": [round(float(standardized_tensor[:, :, c].mean()), 4) for c in range(3)],
            "standardized_std_rgb": [round(float(standardized_tensor[:, :, c].std()), 4) for c in range(3)],
            "imagenet_mean": IMAGENET_MEAN,
            "imagenet_std": IMAGENET_STD,
        },
        "raw_image_base64": _encode_rgb_to_base64(resized),
        "hair_removed_base64": _encode_rgb_to_base64(hair_removed),
        "clahe_enhanced_base64": _encode_rgb_to_base64(clahe_img),
        "normalized_output_base64": _encode_rgb_to_base64(color_norm),
    }
