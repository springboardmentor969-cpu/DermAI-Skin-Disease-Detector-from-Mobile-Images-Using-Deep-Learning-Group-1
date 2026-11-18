
from pathlib import Path
import sys
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps

# Project paths
BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")

# Same preprocessing used during model training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_models():
    res_data = torch.load(
        RESULTS / "dermai_resnet18.pth",
        map_location=DEVICE,
        weights_only=False
    )

    mob_data = torch.load(
        RESULTS / "mobilenet_v3_small.pth",
        map_location=DEVICE,
        weights_only=False
    )

    # Rebuild ResNet18
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(
        resnet.fc.in_features,
        len(res_data["classes"])
    )
    resnet.load_state_dict(res_data["model_state_dict"])
    resnet.eval()

    # Rebuild MobileNetV3-Small
    mobile = models.mobilenet_v3_small(weights=None)
    mobile.classifier[3] = nn.Linear(
        mobile.classifier[3].in_features,
        len(mob_data["classes"])
    )
    mobile.load_state_dict(mob_data["model_state_dict"])
    mobile.eval()

    return (
        resnet,
        res_data["classes"],
        mobile,
        mob_data["classes"]
    )


def generate_gradcam(model, target_layer, image_tensor):
    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

        def gradient_hook(grad):
            gradients.append(grad)

        output.register_hook(gradient_hook)

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad()

        image_tensor = image_tensor.clone().detach()
        image_tensor.requires_grad_(True)

        with torch.enable_grad():
            output = model(image_tensor)
            predicted_index = output.argmax(dim=1).item()

            score = output[0, predicted_index]
            score.backward()

            activation = activations[0]
            gradient = gradients[0]

            weights = gradient.mean(dim=(2, 3), keepdim=True)

            cam = (weights * activation).sum(dim=1)
            cam = F.relu(cam)

            cam = F.interpolate(
                cam.unsqueeze(1),
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            ).squeeze()

            cam = cam.detach().cpu()

            cam_min = cam.min()
            cam_max = cam.max()

            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            probabilities = torch.softmax(output, dim=1)
            score_value = probabilities[0, predicted_index].item()

        return cam, predicted_index, score_value

    finally:
        handle.remove()


def save_visualization(original, cam, output_path):
    # Convert Grad-CAM to a colored heatmap
    grayscale = Image.fromarray(
        (cam.numpy() * 255).astype("uint8")
    )

    heatmap = ImageOps.colorize(
        grayscale,
        black="#101020",
        mid="#ffb000",
        white="#ff3030"
    )

    heatmap = heatmap.resize(original.size)

    overlay = Image.blend(
        original.convert("RGB"),
        heatmap,
        alpha=0.45
    )

    overlay.save(output_path)


def main():
    # Optional image path supplied in terminal
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        image_folder = BASE / "dataset" / "images"

        candidates = []
        for extension in ("*.jpg", "*.jpeg", "*.png"):
            candidates.extend(image_folder.rglob(extension))

        if not candidates:
            raise FileNotFoundError("No dataset images found.")

        image_path = sorted(candidates)[0]

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    original = Image.open(image_path).convert("RGB")
    image_tensor = transform(original).unsqueeze(0).to(DEVICE)

    resnet, res_classes, mobile, mob_classes = load_models()

    model_settings = [
        (
            "ResNet18",
            resnet,
            resnet.layer4[-1],
            res_classes
        ),
        (
            "MobileNetV3-Small",
            mobile,
            mobile.features[-1],
            mob_classes
        )
    ]

    report_rows = []

    print("\nDERMAI - GRAD-CAM EXPLAINABILITY")
    print("Image:", image_path.name)

    for name, model, target_layer, classes in model_settings:
        cam, index, confidence = generate_gradcam(
            model,
            target_layer,
            image_tensor
        )

        predicted_class = classes[index]

        output_path = RESULTS / (
            name.lower().replace("-", "_") + "_gradcam.png"
        )

        save_visualization(
            original,
            cam,
            output_path
        )

        report_rows.append([
            name,
            predicted_class,
            round(confidence, 4),
            str(output_path)
        ])

        print("\nModel:", name)
        print("Predicted class:", predicted_class)
        print("Model score:", round(confidence * 100, 2), "%")
        print("Grad-CAM saved:", output_path)

    report_path = RESULTS / "gradcam_summary.csv"

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model",
            "Predicted_Class",
            "Model_Score",
            "GradCAM_Image"
        ])
        writer.writerows(report_rows)

    print("\nGRAD-CAM COMPLETED")
    print("Summary saved:", report_path)
    print("Educational visualization only; not a medical diagnosis.")


if __name__ == "__main__":
    main()
