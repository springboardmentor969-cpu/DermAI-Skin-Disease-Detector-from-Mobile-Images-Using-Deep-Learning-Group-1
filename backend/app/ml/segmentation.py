from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEGMENTATION_DATASET_URL = "https://challenge.isic-archive.com/2016/"
SEGMENTATION_IMAGE_SIZE = 256


def _write_new_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    with path.open("x", encoding=encoding) as file:
        file.write(content)


def _make_segmentation_splits(
    pair_count: int, seed: int
) -> dict[str, list[int]]:
    order = list(range(pair_count))
    random.Random(seed).shuffle(order)
    test_start = int(pair_count * 0.8)
    validation_start = int(pair_count * 0.7)
    return {
        "train": order[:validation_start],
        "validation": order[validation_start:test_start],
        "test": order[test_start:],
    }


def discover_pairs(dataset_directory: Path) -> list[tuple[Path, Path]]:
    """Find ISBI 2016 image/mask pairs using the dataset's suffix convention."""
    image_directory = dataset_directory / "images" / "ISBI2016_ISIC_Part1_Training_Data"
    mask_directory = (
        dataset_directory / "masks" / "ISBI2016_ISIC_Part1_Training_GroundTruth"
    )
    if not image_directory.is_dir() or not mask_directory.is_dir():
        raise FileNotFoundError(
            "Segmentation dataset must contain images\\ and masks\\ directories. "
            f"Download ISBI 2016 ISIC Part 1 from {SEGMENTATION_DATASET_URL}."
        )
    masks = {
        path.stem.removesuffix("_Segmentation"): path
        for path in mask_directory.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    pairs = [
        (path, masks[path.stem])
        for path in sorted(image_directory.iterdir())
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and path.stem in masks
    ]
    if not pairs:
        raise ValueError("No image/mask pairs were found in the segmentation dataset.")
    return pairs


class LesionSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        indices: list[int],
        training: bool = False,
    ) -> None:
        self.pairs = [pairs[index] for index in indices]
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((SEGMENTATION_IMAGE_SIZE, SEGMENTATION_IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (SEGMENTATION_IMAGE_SIZE, SEGMENTATION_IMAGE_SIZE),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.PILToTensor(),
            ]
        )
        self.training = training

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image_tensor = self.image_transform(image.convert("RGB"))
            mask_tensor = self.mask_transform(mask.convert("L")).float().div(255)
        if self.training and random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
            mask_tensor = torch.flip(mask_tensor, dims=[2])
        return image_tensor, (mask_tensor > 0.5).float()


def dice_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = (torch.sigmoid(logits) > 0.5).float()
    intersection = (prediction * target).sum()
    return float((2 * intersection / (prediction.sum() + target.sum()).clamp_min(1)).item())


def iou_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = torch.sigmoid(logits) > 0.5
    target_bool = target > 0.5
    intersection = (prediction & target_bool).sum()
    union = (prediction | target_bool).sum()
    return float((intersection / union.clamp_min(1)).item())


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class AttentionGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.gate(inputs)


class UNet(nn.Module):
    def __init__(self, attention: bool = False) -> None:
        super().__init__()
        self.attention = attention
        self.down1 = DoubleConv(3, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = DoubleConv(256 + 128, 128)
        self.up2 = DoubleConv(128 + 64, 64)
        self.up1 = DoubleConv(64 + 32, 32)
        self.output = nn.Conv2d(32, 1, 1)
        self.attention3 = AttentionGate(128)
        self.attention2 = AttentionGate(64)
        self.attention1 = AttentionGate(32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = self.down1(inputs)
        second = self.down2(self.pool(first))
        third = self.down3(self.pool(second))
        bottleneck = self.down4(self.pool(third))
        decoded = torch.nn.functional.interpolate(
            bottleneck, size=third.shape[-2:], mode="bilinear", align_corners=False
        )
        skip = self.attention3(third) if self.attention else third
        decoded = self.up3(torch.cat([decoded, skip], dim=1))
        decoded = torch.nn.functional.interpolate(
            decoded, size=second.shape[-2:], mode="bilinear", align_corners=False
        )
        skip = self.attention2(second) if self.attention else second
        decoded = self.up2(torch.cat([decoded, skip], dim=1))
        decoded = torch.nn.functional.interpolate(
            decoded, size=first.shape[-2:], mode="bilinear", align_corners=False
        )
        skip = self.attention1(first) if self.attention else first
        return self.output(self.up1(torch.cat([decoded, skip], dim=1)))


def build_segmentation_model(name: str = "unet") -> UNet:
    if name not in {"unet", "attention_unet"}:
        raise ValueError("Segmentation model must be 'unet' or 'attention_unet'.")
    return UNet(attention=name == "attention_unet")


def train_segmentation(
    dataset_directory: Path,
    output_directory: Path,
    epochs: int = 30,
    batch_size: int = 8,
    seed: int = 42,
    evaluate_attention: bool = False,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    output_directory = output_directory.resolve()
    if output_directory.exists():
        if not output_directory.is_dir():
            raise NotADirectoryError(
                f"Segmentation output path is not a directory: {output_directory}"
            )
        if any(output_directory.iterdir()):
            raise FileExistsError(
                "Segmentation output directory must be new or empty: "
                f"{output_directory}"
            )
    else:
        output_directory.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pairs = discover_pairs(dataset_directory)
    splits = _make_segmentation_splits(len(pairs), seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, Any] = {
        "dataset": "ISBI 2016 ISIC Part 1",
        "dataset_url": SEGMENTATION_DATASET_URL,
        "pair_count": len(pairs),
        "split_counts": {name: len(indices) for name, indices in splits.items()},
        "device": str(device),
        "models": {},
    }
    for model_name in ("unet", "attention_unet") if evaluate_attention else ("unet",):
        loaders = {
            name: DataLoader(
                LesionSegmentationDataset(
                    pairs, indices, training=name == "train"
                ),
                batch_size=batch_size,
                shuffle=name == "train",
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            for name, indices in splits.items()
        }
        model = build_segmentation_model(model_name).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_function = nn.BCEWithLogitsLoss()
        best_dice = -1.0
        history: list[dict[str, float]] = []
        checkpoint = output_directory / f"{model_name}.pth"
        temporary = output_directory / f".{model_name}.pth.partial"
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for images, masks in loaders["train"]:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(images), masks)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.detach()) * images.size(0)
            model.eval()
            validation_dice: list[float] = []
            validation_iou: list[float] = []
            validation_loss = 0.0
            with torch.inference_mode():
                for images, masks in loaders["validation"]:
                    images, masks = images.to(device), masks.to(device)
                    logits = model(images)
                    validation_loss += float(loss_function(logits, masks)) * images.size(0)
                    validation_dice.append(dice_score(logits, masks))
                    validation_iou.append(iou_score(logits, masks))
            dice = float(np.mean(validation_dice))
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss / max(len(splits["train"]), 1),
                    "validation_loss": validation_loss / max(len(splits["validation"]), 1),
                    "validation_dice": dice,
                    "validation_iou": float(np.mean(validation_iou)),
                }
            )
            if dice > best_dice:
                best_dice = dice
                torch.save(model.state_dict(), temporary)
        model.load_state_dict(torch.load(temporary, map_location=device, weights_only=True))
        model.load_state_dict(
            torch.load(temporary, map_location=device, weights_only=True)
        )
        model.eval()
        os.rename(temporary, checkpoint)
        report["models"][model_name] = {
            "weights": checkpoint.name,
            "best_validation_dice": best_dice,
            "history": history,
        }
    selected_model = max(
        report["models"],
        key=lambda name: report["models"][name]["best_validation_dice"],
    )
    selected_result = report["models"][selected_model]
    selected_checkpoint = output_directory / selected_result["weights"]
    selected_instance = build_segmentation_model(selected_model).to(device)
    selected_instance.load_state_dict(
        torch.load(selected_checkpoint, map_location=device, weights_only=True)
    )
    selected_instance.eval()
    test_dice: list[float] = []
    test_iou: list[float] = []
    test_loader = DataLoader(
        LesionSegmentationDataset(pairs, splits["test"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    with torch.inference_mode():
        for images, masks in test_loader:
            logits = selected_instance(images.to(device)).cpu()
            test_dice.append(dice_score(logits, masks))
            test_iou.append(iou_score(logits, masks))
    selected_result["test_dice"] = float(np.mean(test_dice))
    selected_result["test_iou"] = float(np.mean(test_iou))
    _write_new_text(
        output_directory / "selection.json",
        json.dumps(
            {
                "architecture": selected_model,
                "weights": report["models"][selected_model]["weights"],
                "selection_metric": "validation Dice score",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report["selected_model"] = selected_model
    _write_new_text(
        output_directory / "segmentation_report.json",
        json.dumps(report, indent=2),
    )
    lines = [
        "# DermAI lesion segmentation evaluation",
        "",
        "- Dataset: ISBI 2016 ISIC Part 1 (pixel-level lesion masks).",
        f"- Image/mask pairs: {len(pairs)}.",
        f"- Device: {device}.",
        f"- Split sizes: {report['split_counts']}.",
        f"- Selected model: **{selected_model}**, by validation Dice score.",
        "",
        "| Model | Validation Dice | Test Dice | Test IoU |",
        "|---|---:|---:|---:|",
    ]
    for model_name, result in report["models"].items():
        test_values = (
            f"{result['test_dice']:.4f} | {result['test_iou']:.4f}"
            if model_name == selected_model
            else "not evaluated | not evaluated"
        )
        lines.append(
            f"| {model_name} | {result['best_validation_dice']:.4f} | "
            f"{test_values} |"
        )
    _write_new_text(
        output_directory / "segmentation_report.md",
        "\n".join(lines) + "\n",
    )
    return report


def evaluate_segmentation_checkpoint(
    dataset_directory: Path,
    checkpoint_path: Path,
    output_directory: Path,
    seed: int = 42,
    batch_size: int = 8,
    architecture: str = "unet",
) -> dict[str, Any]:
    """Evaluate an existing checkpoint without entering the training loop."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    checkpoint_path = checkpoint_path.resolve()
    output_directory = output_directory.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Segmentation checkpoint was not found: {checkpoint_path}")
    if output_directory.exists():
        if not output_directory.is_dir():
            raise NotADirectoryError(
                f"Segmentation output path is not a directory: {output_directory}"
            )
        existing = {
            "selection.json",
            "segmentation_report.json",
            "segmentation_report.md",
        } & {path.name for path in output_directory.iterdir()}
        if existing:
            raise FileExistsError(
                "Evaluation reports already exist and will not be overwritten: "
                f"{', '.join(sorted(existing))}"
            )
    else:
        output_directory.mkdir(parents=True)

    pairs = discover_pairs(dataset_directory)
    splits = _make_segmentation_splits(len(pairs), seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segmentation_model(architecture).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    loaders = {
        name: DataLoader(
            LesionSegmentationDataset(pairs, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for name, indices in splits.items()
    }

    metrics: dict[str, float] = {}
    with torch.inference_mode():
        for split_name in ("validation", "test"):
            dice_scores: list[float] = []
            iou_scores: list[float] = []
            for images, masks in loaders[split_name]:
                logits = model(images.to(device)).cpu()
                dice_scores.append(dice_score(logits, masks))
                iou_scores.append(iou_score(logits, masks))
            metrics[f"{split_name}_dice"] = float(np.mean(dice_scores))
            metrics[f"{split_name}_iou"] = float(np.mean(iou_scores))

    report: dict[str, Any] = {
        "dataset": "ISBI 2016 ISIC Part 1",
        "dataset_url": SEGMENTATION_DATASET_URL,
        "pair_count": len(pairs),
        "split_counts": {name: len(indices) for name, indices in splits.items()},
        "seed": seed,
        "device": str(device),
        "evaluation_only": True,
        "checkpoint": checkpoint_path.name,
        "selected_model": architecture,
        "models": {
            architecture: {
                "weights": checkpoint_path.name,
                "validation_dice": metrics["validation_dice"],
                "validation_iou": metrics["validation_iou"],
                "test_dice": metrics["test_dice"],
                "test_iou": metrics["test_iou"],
            }
        },
    }
    _write_new_text(
        output_directory / "selection.json",
        json.dumps(
            {
                "architecture": architecture,
                "weights": checkpoint_path.name,
                "selection_metric": "validation Dice score",
            },
            indent=2,
        ),
    )
    _write_new_text(
        output_directory / "segmentation_report.json",
        json.dumps(report, indent=2),
    )
    _write_new_text(
        output_directory / "segmentation_report.md",
        "\n".join(
            [
                "# DermAI lesion segmentation evaluation",
                "",
                "- Dataset: ISBI 2016 ISIC Part 1 (pixel-level lesion masks).",
                "- Evaluation mode: existing checkpoint; no training was run.",
                f"- Checkpoint: `{checkpoint_path}`.",
                f"- Image/mask pairs: {len(pairs)}.",
                f"- Split sizes: {report['split_counts']}.",
                f"- Validation Dice: {metrics['validation_dice']:.4f}.",
                f"- Validation IoU: {metrics['validation_iou']:.4f}.",
                f"- Test Dice: {metrics['test_dice']:.4f}.",
                f"- Test IoU: {metrics['test_iou']:.4f}.",
            ]
        )
        + "\n",
    )
    return report


def create_overlay(image: Image.Image, mask: torch.Tensor) -> Image.Image:
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 40, 40, 0))
    mask_image = Image.fromarray(
        (mask.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    ).resize(image.size, Image.Resampling.NEAREST)
    dilated = mask_image.filter(ImageFilter.MaxFilter(5))
    eroded = mask_image.filter(ImageFilter.MinFilter(5))
    alpha = ImageChops.subtract(dilated, eroded)
    overlay.putalpha(alpha.point(lambda value: int(value * 0.45)))
    return Image.alpha_composite(image, overlay).convert("RGB")
