import torch.nn as nn
from torchvision import models


def create_densenet121(num_classes=6, pretrained=False):
    """
    Create the DermAI DenseNet-121 classification model.

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
        DenseNet-121 model with a 6-class classification head.
    """

    if pretrained:
        weights = models.DenseNet121_Weights.DEFAULT
    else:
        weights = None

    model = models.densenet121(weights=weights)

    input_features = model.classifier.in_features

    model.classifier = nn.Linear(
        input_features,
        num_classes
    )

    return model