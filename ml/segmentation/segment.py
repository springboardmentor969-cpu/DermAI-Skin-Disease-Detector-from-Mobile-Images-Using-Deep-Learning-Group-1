"""
Lesion segmentation inference and Region of Interest (ROI) extraction.

Provides `segment_lesion()` which:
- Accepts an image (file path, PIL Image, or numpy array).
- Predicts the lesion boundary mask with a trained U-Net.
- Computes the lesion bounding box with an adjustable margin.
- Crops the lesion ROI from the original image.
- If the mask is empty or very small (below threshold), returns the full image
  and a flag indicating that segmentation failed.
"""

import sys
from pathlib import Path
from typing import Union, Tuple, Optional, NamedTuple
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.segmentation.model import build_segmentation_unet

# Global cached model to avoid re-loading weights on every single call
_CACHED_MODEL: Optional[nn.Module] = None
_CACHED_CKPT_PATH: Optional[str] = None


class SegmentationResult(NamedTuple):
    """
    Result container that supports both named attributes and 4-element tuple unpacking:
    `mask, bbox, cropped_roi, success = segment_lesion(...)`
    """
    mask: np.ndarray                      # Binary uint8 mask (0 or 255) with original (H, W)
    bbox: Tuple[int, int, int, int]       # (x_min, y_min, x_max, y_max)
    cropped_roi: np.ndarray               # RGB cropped ROI (or full image if failed)
    success: bool                         # False if mask is empty or too small, True otherwise


def load_segmentation_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """
    Loads or retrieves a cached U-Net segmentation model.
    """
    global _CACHED_MODEL, _CACHED_CKPT_PATH

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Resolve default checkpoint path if not provided
    if checkpoint_path is None:
        candidate_paths = [
            PROJECT_ROOT / "ml" / "experiments" / "segmentation" / "best_model.pth",
            PROJECT_ROOT / "ml" / "segmentation" / "best_model.pth",
        ]
        for p in candidate_paths:
            if p.is_file():
                checkpoint_path = p
                break

    ckpt_str = str(checkpoint_path) if checkpoint_path else "scratch"

    if _CACHED_MODEL is not None and _CACHED_CKPT_PATH == ckpt_str:
        return _CACHED_MODEL.to(device)

    # Build model architecture
    model = build_segmentation_unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if checkpoint_path is None else None,
        in_channels=3,
        classes=1,
    )

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            # Try raw dict
            try:
                model.load_state_dict(ckpt)
            except Exception:
                pass
        print(f"[INFO] Loaded segmentation model checkpoint: {checkpoint_path}")

    model.to(device)
    model.eval()

    _CACHED_MODEL = model
    _CACHED_CKPT_PATH = ckpt_str
    return model


def segment_lesion(
    image: Union[str, Path, Image.Image, np.ndarray],
    model: Optional[nn.Module] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    threshold: float = 0.5,
    margin: float = 0.10,
    min_area_pixels: int = 100,
    min_area_ratio: float = 0.001,
    img_size: int = 256,
) -> SegmentationResult:
    """
    Segments a skin lesion from an input image, returning the mask, bounding box,
    and cropped ROI.

    Args:
        image: File path (str/Path), PIL Image, or numpy array (RGB, uint8).
        model: Optional pre-loaded U-Net model.
        checkpoint_path: Optional path to weights.
        device: Device to run inference on.
        threshold: Probability threshold for binarizing predicted mask.
        margin: Fraction of bbox dimension to expand on each side (default 0.10, 10%).
        min_area_pixels: Minimum lesion pixel area; below this, segmentation is marked failed.
        min_area_ratio: Minimum fraction of total image area required.
        img_size: Resize dimension for model inference (default 256).

    Returns:
        SegmentationResult (named tuple):
            - mask: np.ndarray uint8 (H, W), values {0, 255}
            - bbox: Tuple (x_min, y_min, x_max, y_max)
            - cropped_roi: np.ndarray (H_roi, W_roi, 3) RGB
            - success: bool (False if empty or very small, True otherwise)
    """
    # ── 1. Prepare image as RGB numpy array ─────────────────────────────────
    if isinstance(image, (str, Path)):
        img_p = Path(image)
        if not img_p.is_file():
            raise FileNotFoundError(f"Image not found at {img_p}")
        bgr = cv2.imread(str(img_p))
        if bgr is None:
            raise ValueError(f"Could not decode image at {img_p}")
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            img_rgb = image.copy()
        elif image.ndim == 3 and image.shape[2] == 4:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError(f"Unsupported numpy image shape: {image.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    orig_h, orig_w = img_rgb.shape[:2]
    total_pixels = orig_h * orig_w

    # ── 2. Load model ─────────────────────────────────────────────────────────
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if model is None:
        model = load_segmentation_model(checkpoint_path=checkpoint_path, device=device)
    else:
        model.to(device)
        model.eval()

    # ── 3. Preprocess for inference (Resize 256x256 + ImageNet Normalize) ────
    resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor_input = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # ── 4. Forward inference ──────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (img_size, img_size)

    # ── 5. Resize prediction back to original dimensions ─────────────────────
    prob_full = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask_binary = (prob_full >= threshold).astype(np.uint8) * 255

    # ── 6. Area check and bounding box computation ───────────────────────────
    lesion_area = int((mask_binary > 0).sum())
    min_required_pixels = max(min_area_pixels, int(min_area_ratio * total_pixels))

    # If mask is empty or very small: segmentation failed!
    if lesion_area < min_required_pixels:
        full_bbox = (0, 0, orig_w, orig_h)
        return SegmentationResult(
            mask=mask_binary,
            bbox=full_bbox,
            cropped_roi=img_rgb,
            success=False
        )

    # Find bounding box from positive mask pixels
    y_indices, x_indices = np.where(mask_binary > 0)
    x_min, x_max = int(x_indices.min()), int(x_indices.max())
    y_min, y_max = int(y_indices.min()), int(y_indices.max())

    # Add margin around bounding box
    bbox_w = x_max - x_min + 1
    bbox_h = y_max - y_min + 1
    pad_x = int(round(bbox_w * margin))
    pad_y = int(round(bbox_h * margin))

    crop_x_min = max(0, x_min - pad_x)
    crop_y_min = max(0, y_min - pad_y)
    crop_x_max = min(orig_w, x_max + pad_x + 1)
    crop_y_max = min(orig_h, y_max + pad_y + 1)

    cropped_roi = img_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    return SegmentationResult(
        mask=mask_binary,
        bbox=(crop_x_min, crop_y_min, crop_x_max, crop_y_max),
        cropped_roi=cropped_roi,
        success=True
    )


def overlay_mask_on_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.40,
) -> np.ndarray:
    """
    Renders a translucent color overlay of the lesion mask and optional bounding box
    onto the RGB image.
    """
    overlay = image_rgb.copy()
    colored_mask = np.zeros_like(image_rgb)
    colored_mask[mask > 0] = color

    # Blend
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)

    # Draw contour border
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)

    # Draw bbox if provided
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 215, 0), 2)

    return overlay


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Segment lesion in an input image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained U-Net checkpoint.")
    parser.add_argument("--output_dir", type=str, default="ml/experiments/segmentation/demo", help="Directory to save output.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation probability threshold.")
    parser.add_argument("--margin", type=float, default=0.10, help="ROI margin fraction.")
    args = parser.parse_args()

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    result = segment_lesion(
        image=args.image,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        margin=args.margin,
    )

    orig_bgr = cv2.imread(args.image)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # Save mask
    cv2.imwrite(str(out_p / "mask.png"), result.mask)
    # Save cropped ROI
    cv2.imwrite(str(out_p / "cropped_roi.png"), cv2.cvtColor(result.cropped_roi, cv2.COLOR_RGB2BGR))
    # Save visual overlay
    vis = overlay_mask_on_image(orig_rgb, result.mask, bbox=result.bbox)
    cv2.imwrite(str(out_p / "overlay.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"[RESULT] Segmentation Success: {result.success}")
    print(f"[RESULT] Bounding Box        : {result.bbox}")
    print(f"[RESULT] Cropped ROI Shape   : {result.cropped_roi.shape}")
    print(f"[SAVED] Artifacts written to : {out_p}")

