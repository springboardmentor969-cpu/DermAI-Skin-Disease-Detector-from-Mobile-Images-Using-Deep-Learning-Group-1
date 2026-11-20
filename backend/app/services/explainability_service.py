from __future__ import annotations

import base64
import io
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.ml.classification import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE


def _normalize_heatmap(cam: np.ndarray) -> np.ndarray:
    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max > cam_min:
        return (cam - cam_min) / (cam_max - cam_min)
    return np.zeros_like(cam)


def generate_gradcam_explanation(
    model: torch.nn.Module,
    architecture: str,
    image_bytes: bytes,
    target_class_index: int | None = None,
    lesion_mask_bytes: bytes | None = None,
) -> dict[str, Any]:
    """
    Generates explainable AI visualizations (Grad-CAM / Attention Maps) for
    skin lesion classification predictions.
    """
    model.eval()

    # Determine target layer according to architecture
    target_layer: torch.nn.Module | None = None
    is_transformer = "vit" in architecture.lower() or "swin" in architecture.lower()

    if not is_transformer:
        if hasattr(model, "layer4"):  # ResNet
            target_layer = model.layer4[-1]
        elif hasattr(model, "features"):  # MobileNet / EfficientNet / DenseNet
            target_layer = model.features[-1]
        else:
            # Fallback to the last sequential child module
            modules = list(model.children())
            if modules:
                target_layer = modules[-2] if len(modules) > 1 else modules[-1]

    # Preprocess image
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = pil_image.size  # (W, H)
    pil_resized = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    rgb_np = np.array(pil_resized)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    input_tensor = transform(pil_resized).unsqueeze(0)
    input_tensor.requires_grad = True

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    if not is_transformer and target_layer is not None:
        def forward_hook(module: torch.nn.Module, inp: Any, out: torch.Tensor) -> None:
            activations.append(out)

        def backward_hook(module: torch.nn.Module, grad_inp: Any, grad_out: tuple[torch.Tensor, ...]) -> None:
            gradients.append(grad_out[0])

        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        logits = model(input_tensor)
        if target_class_index is None:
            target_class_index = int(logits.argmax(dim=1).item())

        model.zero_grad()
        score = logits[0, target_class_index]
        score.backward(retain_graph=True)

        fwd_handle.remove()
        bwd_handle.remove()

        if activations and gradients:
            act = activations[0].detach()[0]  # [C, H, W]
            grad = gradients[0].detach()[0]  # [C, H, W]
            weights = grad.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
            cam = torch.relu((weights * act).sum(dim=0)).cpu().numpy()
        else:
            cam = np.zeros((7, 7), dtype=np.float32)

        method_name = "Grad-CAM (Gradient-weighted Class Activation Mapping)"
    else:
        # Vision Transformer attention-based explainability
        # Use gradient of output w.r.t input spatial features / tokens
        logits = model(input_tensor)
        if target_class_index is None:
            target_class_index = int(logits.argmax(dim=1).item())

        model.zero_grad()
        score = logits[0, target_class_index]
        score.backward()

        if input_tensor.grad is not None:
            grad = input_tensor.grad.detach()[0]  # [3, H, W]
            saliency = grad.abs().max(dim=0)[0].cpu().numpy()
            # Smooth saliency map to produce clean attention regions
            cam = cv2.GaussianBlur(saliency, (15, 15), 0)
        else:
            cam = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        method_name = "Transformer Attention Gradient Saliency Map"

    # Normalize CAM to [0, 1]
    norm_cam = _normalize_heatmap(cam)
    norm_cam_resized = cv2.resize(norm_cam, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    norm_cam_resized = np.clip(norm_cam_resized, 0.0, 1.0)

    # Calculate metrics:
    peak_y, peak_x = np.unravel_index(np.argmax(norm_cam_resized), norm_cam_resized.shape)
    coverage_fraction = float(np.mean(norm_cam_resized > 0.40)) * 100.0

    # Colorize heatmap (JET colormap)
    heatmap_uint8 = np.uint8(255 * norm_cam_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend with original RGB: alpha 0.40 heatmap + 0.60 original
    blended = cv2.addWeighted(rgb_np, 0.60, heatmap_colored_rgb, 0.40, 0)

    # Encode images to base64
    def _to_base64_url(arr: np.ndarray) -> str:
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    heatmap_url = _to_base64_url(heatmap_colored_rgb)
    overlay_url = _to_base64_url(blended)

    # Check lesion mask overlap if provided
    lesion_overlap_score: float | None = None
    if lesion_mask_bytes:
        try:
            mask_pil = Image.open(io.BytesIO(lesion_mask_bytes)).convert("L")
            mask_resized = np.array(mask_pil.resize((IMAGE_SIZE, IMAGE_SIZE))) > 127
            cam_binary = norm_cam_resized > 0.40
            intersection = np.logical_and(cam_binary, mask_resized).sum()
            union = np.logical_or(cam_binary, mask_resized).sum()
            if union > 0:
                lesion_overlap_score = round(float(intersection / union), 4)
        except Exception:
            pass

    return {
        "status": "available",
        "method": method_name,
        "heatmap_data_url": heatmap_url,
        "overlay_data_url": overlay_url,
        "peak_attention_coords": [int(peak_x), int(peak_y)],
        "attention_coverage_percent": round(coverage_fraction, 2),
        "lesion_overlap_score": lesion_overlap_score,
        "interpretation": (
            f"The AI model's attention concentrated {coverage_fraction:.1f}% of key visual "
            f"features around lesion region coordinates ({peak_x}, {peak_y})."
        ),
    }

