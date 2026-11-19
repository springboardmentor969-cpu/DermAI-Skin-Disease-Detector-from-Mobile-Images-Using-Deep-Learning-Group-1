"""
Grad-CAM Explainability Module for DermAI.

Generates gradient-weighted class activation maps (Grad-CAM) to visualize
and explain model predictions on skin lesion images using pytorch-grad-cam.
Supports CNNs (ResNet, EfficientNet, DenseNet, MobileNet) and Vision Transformers (ViT, Swin).
"""

import io
import logging
from typing import Dict, Any, Optional, Union, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

logger = logging.getLogger("dermai.gradcam")

# Standard ImageNet normalization stats used across DermAI
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_CLASSES: Dict[str, int] = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}


def reshape_transform_vit(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reshape transform for standard Vision Transformer models (ViT).
    Discards the [CLS] token and reshapes 1D patch tokens to a 2D spatial grid (B, C, H, W).
    """
    # tensor shape: (batch_size, num_tokens, embed_dim)
    num_patches = tensor.shape[1] - 1
    height = width = int(num_patches ** 0.5)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring channels to dimension 1: (B, C, H, W)
    return result.permute(0, 3, 1, 2)


def reshape_transform_swin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reshape transform for Swin Transformer models.
    Permutes (B, H, W, C) or (B, N, C) into (B, C, H, W).
    """
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 3:
        b, n, c = tensor.shape
        h = w = int(n ** 0.5)
        return tensor.view(b, h, w, c).permute(0, 3, 1, 2)
    return tensor


def detect_target_layers(model: nn.Module) -> Tuple[List[nn.Module], Optional[callable]]:
    """
    Auto-detects the optimal target layer and reshape transform for Grad-CAM.
    Supports ViT, Swin, ResNet, DenseNet, EfficientNet, MobileNet, and generic CNNs.
    """
    # Unwrap DataParallel / DistributedDataParallel if wrapped
    base_model = model.module if hasattr(model, "module") else model
    model_cls_name = base_model.__class__.__name__.lower()

    # 1. Vision Transformer (ViT)
    if ("visiontransformer" in model_cls_name or "vit" in model_cls_name) and hasattr(base_model, "blocks"):
        last_block = base_model.blocks[-1]
        target_layer = getattr(last_block, "norm1", last_block)
        logger.debug(f"[GradCAM] Auto-detected ViT target layer: {target_layer}")
        return [target_layer], reshape_transform_vit

    # 2. Swin Transformer
    if "swin" in model_cls_name and hasattr(base_model, "layers"):
        last_stage = base_model.layers[-1]
        if hasattr(last_stage, "blocks") and len(last_stage.blocks) > 0:
            last_block = last_stage.blocks[-1]
            target_layer = getattr(last_block, "norm1", last_block)
            logger.debug(f"[GradCAM] Auto-detected Swin target layer: {target_layer}")
            return [target_layer], reshape_transform_swin

    # 3. ResNet family (layer4)
    if hasattr(base_model, "layer4") and len(base_model.layer4) > 0:
        target_layer = base_model.layer4[-1]
        logger.debug(f"[GradCAM] Auto-detected ResNet target layer: {target_layer}")
        return [target_layer], None

    # 4. DenseNet family (features.norm5 or features.denseblock4)
    if hasattr(base_model, "features"):
        if hasattr(base_model.features, "norm5"):
            target_layer = base_model.features.norm5
        elif hasattr(base_model.features, "denseblock4"):
            target_layer = base_model.features.denseblock4
        else:
            target_layer = list(base_model.features.children())[-1]
        logger.debug(f"[GradCAM] Auto-detected DenseNet target layer: {target_layer}")
        return [target_layer], None

    # 5. EfficientNet / MobileNet family (conv_head or act2)
    if hasattr(base_model, "conv_head") and base_model.conv_head is not None:
        logger.debug(f"[GradCAM] Auto-detected conv_head target layer: {base_model.conv_head}")
        return [base_model.conv_head], None
    if hasattr(base_model, "act2") and base_model.act2 is not None:
        logger.debug(f"[GradCAM] Auto-detected act2 target layer: {base_model.act2}")
        return [base_model.act2], None

    # 6. Fallback: Search backwards for the last Conv2d layer in any CNN
    conv_layers = [m for m in base_model.modules() if isinstance(m, nn.Conv2d)]
    if conv_layers:
        logger.debug(f"[GradCAM] Auto-detected last Conv2d target layer: {conv_layers[-1]}")
        return [conv_layers[-1]], None

    raise ValueError(
        f"Unable to auto-detect target convolutional or transformer layer for model: {base_model.__class__.__name__}"
    )


def generate_gradcam(
    model: nn.Module,
    image_tensor: Union[torch.Tensor, np.ndarray],
    target_class: Optional[Union[int, str]] = None,
    original_image: Optional[Union[np.ndarray, PILImage.Image]] = None,
) -> Dict[str, Any]:
    """
    Generates Grad-CAM class activation map and blended PNG overlay.

    Args:
        model: Trained PyTorch model (CNN or ViT/Swin).
        image_tensor: Preprocessed tensor of shape (1, 3, H, W) or (3, H, W).
        target_class: Target class index (int) or class code (str, e.g. 'mel').
                      If None, uses the predicted class (highest logit).
        original_image: Optional unnormalized original image for overlay blending.
                        If None, image_tensor is unnormalized with ImageNet statistics.

    Returns:
        Dictionary containing:
            - heatmap: 2D numpy array (H, W) with values normalized to [0, 1]
            - overlay_bytes: PNG bytes of heatmap blended on original image with jet colormap (alpha=0.4)
            - target_class: integer class index explained
    """
    # 1. Format input tensor
    if isinstance(image_tensor, np.ndarray):
        tensor = torch.from_numpy(image_tensor).float()
    else:
        tensor = image_tensor.clone().float()

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    # Determine device from model parameters
    try:
        device = next(model.parameters()).device
    except (StopIteration, Exception):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor = tensor.to(device)

    # 2. Determine target class
    target_idx: int
    if target_class is None:
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            target_idx = int(torch.argmax(logits, dim=-1).item())
    elif isinstance(target_class, str):
        target_idx = DEFAULT_CLASSES.get(target_class.lower(), 0)
    else:
        target_idx = int(target_class)

    targets = [ClassifierOutputTarget(target_idx)]

    # 3. Detect target layer and transform
    target_layers, reshape_transform = detect_target_layers(model)

    # 4. Generate CAM heatmap
    model.eval()
    with GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    ) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)
        heatmap = grayscale_cam[0, :]  # Shape: (H, W), values in [0, 1]

    heatmap = np.nan_to_num(heatmap, nan=0.0)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    # 5. Prepare original RGB image for blending
    orig_rgb: np.ndarray
    if original_image is not None:
        if isinstance(original_image, PILImage.Image):
            orig_rgb = np.array(original_image.convert("RGB"))
        else:
            orig_rgb = np.array(original_image)
    else:
        # Unnormalize ImageNet tensor: (C, H, W) -> (H, W, C)
        img_np = tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        unnorm = img_np * IMAGENET_STD + IMAGENET_MEAN
        orig_rgb = np.clip(unnorm * 255.0, 0, 255).astype(np.uint8)

    # Ensure dimensions match
    orig_h, orig_w = orig_rgb.shape[:2]
    if (heatmap.shape[0], heatmap.shape[1]) != (orig_h, orig_w):
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        heatmap_resized = heatmap

    # 6. Apply JET colormap and alpha blend (alpha=0.4)
    heatmap_u8 = np.uint8(255 * heatmap_resized)
    colored_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    # Pure heatmap: jet colormap on black background (suppress negligible activations to pure black)
    heatmap_black_bgr = colored_bgr.copy()
    heatmap_black_bgr[heatmap_resized < 0.02] = 0

    if orig_rgb.shape[2] == 3:
        orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    else:
        orig_bgr = orig_rgb

    alpha = 0.4
    blended_bgr = np.clip(
        alpha * colored_bgr.astype(np.float32) + (1.0 - alpha) * orig_bgr.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    # Encode overlay as PNG bytes
    success, encoded_buf = cv2.imencode(".png", blended_bgr)
    if not success:
        raise RuntimeError("Failed to encode Grad-CAM overlay image to PNG.")
    overlay_bytes = encoded_buf.tobytes()

    # Encode pure heatmap as PNG bytes
    success_heat, encoded_heat = cv2.imencode(".png", heatmap_black_bgr)
    if not success_heat:
        raise RuntimeError("Failed to encode Grad-CAM pure heatmap image to PNG.")
    heatmap_bytes = encoded_heat.tobytes()

    return {
        "heatmap": heatmap,
        "overlay_bytes": overlay_bytes,
        "heatmap_bytes": heatmap_bytes,
        "target_class": target_idx,
    }
