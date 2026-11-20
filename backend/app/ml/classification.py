from __future__ import annotations

from typing import Any

MODEL_NAMES = (
    "mobilenet_v2",
    "efficientnet_b0",
    "resnet50",
    "densenet121",
    "vit_b_16",
)
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(name: str, class_count: int, pretrained: bool = False) -> Any:
    import torch
    from torchvision.models import (
        DenseNet121_Weights,
        EfficientNet_B0_Weights,
        MobileNet_V2_Weights,
        ResNet50_Weights,
        ViT_B_16_Weights,
        densenet121,
        efficientnet_b0,
        mobilenet_v2,
        resnet50,
        vit_b_16,
    )

    if name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(input_features, class_count)
        feature_layers = model.features
    elif name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(input_features, class_count)
        feature_layers = model.features
    elif name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        input_features = model.fc.in_features
        model.fc = torch.nn.Linear(input_features, class_count)
        feature_layers = model.layer4
    elif name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        input_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(input_features, class_count)
        feature_layers = model.features
    elif name == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
        input_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(input_features, class_count)
        feature_layers = model.encoder.layers[-1]
    else:
        raise ValueError(f"Unsupported classifier architecture: {name}")

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in feature_layers.parameters():
        parameter.requires_grad = True
    if hasattr(model, "classifier"):
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    elif hasattr(model, "fc"):
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    elif hasattr(model, "heads"):
        for parameter in model.heads.parameters():
            parameter.requires_grad = True

    return model
