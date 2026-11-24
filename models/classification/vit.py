import torch.nn as nn
from torchvision import models


def create_vit(num_classes=6, pretrained=False):
    """
    Create the DermAI Vision Transformer (ViT-B/16) classification model.

    Classes:
        ACK, BCC, MEL, NEV, SCC, SEK

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use ImageNet pretrained weights.

    Returns:
        torch.nn.Module: ViT-B/16 classification model.
    """

    if pretrained:
        weights = models.ViT_B_16_Weights.DEFAULT
    else:
        weights = None

    model = models.vit_b_16(weights=weights)

    # Replace the original ImageNet classification head
    input_features = model.heads.head.in_features

    model.heads.head = nn.Linear(
        input_features,
        num_classes
    )

    return model