"""
Model architecture definition for DermAI Lesion Segmentation.

Uses U-Net with a pretrained ResNet34 encoder from segmentation_models_pytorch (SMP).
"""

from typing import Optional
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_segmentation_unet(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """
    Constructs a U-Net model with a pretrained encoder.

    Args:
        encoder_name: Backbone encoder name (default 'resnet34').
        encoder_weights: Pretrained weights (e.g. 'imagenet' or None).
        in_channels: Number of input channels (3 for RGB).
        classes: Number of output masks (1 for binary lesion segmentation).

    Returns:
        torch.nn.Module: U-Net model returning raw logits.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  # Output raw logits
    )
    return model
