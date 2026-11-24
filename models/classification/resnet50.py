import torch.nn as nn
from torchvision import models


def create_resnet50(num_classes=6, pretrained=False):
    """
    Create the DermAI ResNet-50 classification model.

    Parameters
    ----------
    num_classes : int
        Number of DermAI disease classes.
    pretrained : bool
        If True, load ImageNet pretrained weights.
        For DermAI inference, this should normally remain False because
        the trained DermAI checkpoint is loaded separately.

    Returns
    -------
    torch.nn.Module
        ResNet-50 model with a 6-class classification head.
    """

    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet50(weights=weights)

    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        num_classes
    )

    return model