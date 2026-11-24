"""
DermAI Grad-CAM Explainable AI

Generates Grad-CAM visualizations for the trained DermAI ResNet-50 model.

Target layer:
    model.layer4[-1].conv3

This uses the trained DermAI checkpoint through the existing
backend.services.inference model-loading service.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image

from backend.services.inference import (
    load_model,
    INFERENCE_TRANSFORM,
    CLASS_NAMES,
    DEVICE,
)


# ============================================================
# 1. PATHS
# ============================================================

DERMAI_ROOT = Path(__file__).resolve().parents[1]

GRADCAM_OUTPUT_DIR = DERMAI_ROOT / "outputs" / "gradcam"

GRADCAM_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. GRAD-CAM CLASS
# ============================================================

class GradCAM:
    """
    Grad-CAM implementation for a CNN classification model.

    The method:
    1. Captures activations from the target convolution layer.
    2. Captures gradients flowing through that layer.
    3. Computes channel weights using global average pooling.
    4. Creates a class-specific activation map.
    5. Upsamples the map to image size.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self._forward_hook
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def _forward_hook(self, module, input_data, output):
        self.activations = output

    # --------------------------------------------------------
    # Backward hook
    # --------------------------------------------------------

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    # --------------------------------------------------------
    # Generate CAM
    # --------------------------------------------------------

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM for one image.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape: [1, 3, 224, 224]

        target_class : int or None
            Class index to explain.
            If None, explains the model's predicted class.

        Returns
        -------
        cam : numpy.ndarray
            Normalized Grad-CAM heatmap in range [0, 1].

        predicted_class : int
            Predicted class index.

        probabilities : numpy.ndarray
            Class probabilities.
        """

        self.model.zero_grad(set_to_none=True)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = self.model(input_tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = int(
            torch.argmax(
                probabilities,
                dim=1
            ).item()
        )

        # ----------------------------------------------------
        # Select target class
        # ----------------------------------------------------

        if target_class is None:
            target_class = predicted_class

        if target_class < 0 or target_class >= len(CLASS_NAMES):
            raise ValueError(
                f"Invalid target class: {target_class}"
            )

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        target_score = output[
            0,
            target_class
        ]

        target_score.backward()

        # ----------------------------------------------------
        # Validate hooks
        # ----------------------------------------------------

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # ----------------------------------------------------
        # Get activations and gradients
        # ----------------------------------------------------

        activations = self.activations
        gradients = self.gradients

        # Shape:
        # [1, channels, height, width]

        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = torch.mean(
            gradients,
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # Weighted combination
        # ----------------------------------------------------

        cam = torch.sum(
            weights * activations,
            dim=1,
            keepdim=True
        )

        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        cam = F.relu(cam)

        # ----------------------------------------------------
        # Resize CAM to input image size
        # ----------------------------------------------------

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        # ----------------------------------------------------
        # Remove batch/channel dimensions
        # ----------------------------------------------------

        cam = cam.squeeze(
            0,
            1
        )

        # ----------------------------------------------------
        # Normalize to [0, 1]
        # ----------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        cam = cam.detach().cpu().numpy()

        probabilities = (
            probabilities[0]
            .detach()
            .cpu()
            .numpy()
        )

        return (
            cam,
            predicted_class,
            probabilities
        )


# ============================================================
# 3. IMAGE PREPARATION
# ============================================================

def prepare_image(image_path):
    """
    Load and preprocess an image using the exact
    DermAI inference preprocessing pipeline.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = INFERENCE_TRANSFORM(
        image
    )

    input_tensor = input_tensor.unsqueeze(
        0
    )

    input_tensor = input_tensor.to(
        DEVICE
    )

    return image, input_tensor


# ============================================================
# 4. CREATE OVERLAY
# ============================================================

def create_gradcam_visualization(
    original_image,
    cam,
    predicted_class,
    confidence,
    output_path,
):
    """
    Create and save:

    Original image
    + Grad-CAM heatmap
    + Grad-CAM overlay
    """

    # --------------------------------------------------------
    # Resize CAM to original image dimensions
    # --------------------------------------------------------

    image_width, image_height = original_image.size

    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    )

    cam_image = cam_image.resize(
        (image_width, image_height),
        Image.Resampling.BILINEAR
    )

    cam_array = np.asarray(
        cam_image
    ) / 255.0

    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    heatmap = plt.get_cmap(
        "jet"
    )(cam_array)[..., :3]

    heatmap = np.uint8(
        heatmap * 255
    )

    heatmap_image = Image.fromarray(
        heatmap
    ).convert("RGB")

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = Image.blend(
        original_image,
        heatmap_image,
        alpha=0.45
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        original_image
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis("off")

    axes[1].imshow(
        heatmap_image
    )

    axes[1].set_title(
        "Grad-CAM Heatmap"
    )

    axes[1].axis("off")

    axes[2].imshow(
        overlay
    )

    axes[2].set_title(
        f"Grad-CAM Overlay\n"
        f"{CLASS_NAMES[predicted_class]} "
        f"({confidence * 100:.2f}%)"
    )

    axes[2].axis("off")

    plt.tight_layout()

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# 5. MAIN GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(
    image_path,
    output_filename=None,
    target_class=None,
):
    """
    Generate a Grad-CAM explanation for a DermAI image.

    Parameters
    ----------
    image_path : str or Path
        Input dermatological image.

    output_filename : str or None
        Output visualization filename.

    target_class : int or None
        Class to explain.
        None = predicted class.
    """

    print("=" * 60)
    print("DERMAI GRAD-CAM EXPLAINABLE AI")
    print("=" * 60)

    print(
        f"Image: {image_path}"
    )

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load trained ResNet-50
    # --------------------------------------------------------

    model = load_model(
        "resnet50"
    )

    model.eval()

    # --------------------------------------------------------
    # Target layer
    # --------------------------------------------------------

    target_layer = model.layer4[-1].conv3

    print(
        "Target layer: model.layer4[-1].conv3"
    )

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    original_image, input_tensor = prepare_image(
        image_path
    )

    # --------------------------------------------------------
    # Create Grad-CAM object
    # --------------------------------------------------------

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer
    )

    try:

        cam, predicted_class, probabilities = (
            gradcam.generate(
                input_tensor=input_tensor,
                target_class=target_class
            )
        )

    finally:

        gradcam.remove_hooks()

    # --------------------------------------------------------
    # Prediction information
    # --------------------------------------------------------

    predicted_condition = CLASS_NAMES[
        predicted_class
    ]

    confidence = float(
        probabilities[predicted_class]
    )

    print(
        f"Predicted class: {predicted_condition}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Output filename
    # --------------------------------------------------------

    if output_filename is None:

        image_name = Path(
            image_path
        ).stem

        output_filename = (
            f"{image_name}_gradcam.png"
        )

    output_path = (
        GRADCAM_OUTPUT_DIR
        / output_filename
    )

    # --------------------------------------------------------
    # Create visualization
    # --------------------------------------------------------

    saved_path = create_gradcam_visualization(
        original_image=original_image,
        cam=cam,
        predicted_class=predicted_class,
        confidence=confidence,
        output_path=output_path,
    )

    print(
        f"Grad-CAM saved to:\n{saved_path}"
    )

    print("=" * 60)

    return {
        "condition": predicted_condition,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[index]: float(
                probabilities[index]
            )
            for index in range(
                len(CLASS_NAMES)
            )
        },
        "output_path": str(
            saved_path
        ),
    }