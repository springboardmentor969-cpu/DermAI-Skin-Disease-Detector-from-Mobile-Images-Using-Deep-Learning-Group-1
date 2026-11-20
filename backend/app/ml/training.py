from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from app.ml.classification import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    MODEL_NAMES,
    build_model,
)

SEED = 42
IMAGE_DIRECTORIES = ("HAM10000_images_part_1", "HAM10000_images_part_2")


def load_records(project_root: Path) -> list[dict[str, str | Path]]:
    metadata_path = project_root / "HAM10000_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"HAM10000 metadata was not found at {metadata_path}.")

    metadata = pd.read_csv(metadata_path)
    required = {"image_id", "lesion_id", "dx"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"HAM10000 metadata must include {sorted(required)}.")
    if metadata.empty or metadata[list(required)].isna().any().any():
        raise ValueError("HAM10000 metadata contains missing required labels or IDs.")

    image_paths: dict[str, Path] = {}
    for directory_name in IMAGE_DIRECTORIES:
        image_directory = project_root / directory_name
        if not image_directory.is_dir():
            raise FileNotFoundError(f"HAM10000 image directory is missing: {image_directory}")
        for path in image_directory.glob("*.jpg"):
            if path.stem in image_paths:
                raise ValueError(f"Duplicate image ID is present in the image folders: {path.stem}")
            image_paths[path.stem] = path

    records: list[dict[str, str | Path]] = []
    for row in metadata.itertuples(index=False):
        image_id = str(row.image_id)
        image_path = image_paths.get(image_id)
        if image_path is None:
            raise FileNotFoundError(f"No JPEG file found for metadata image {image_id}.")
        with image_path.open("rb") as image_file:
            image_hash = hashlib.file_digest(image_file, "sha256").hexdigest()
        records.append(
            {
                "image_id": image_id,
                "lesion_id": str(row.lesion_id),
                "label": str(row.dx),
                "path": image_path,
                "image_hash": image_hash,
            }
        )
    if len(records) != len(image_paths):
        raise ValueError("Image files and metadata rows do not have matching counts.")
    return records


def make_grouped_splits(
    records: list[dict[str, str | Path]], seed: int = SEED
) -> dict[str, list[int]]:
    labels = np.asarray([str(record["label"]) for record in records])
    parents: dict[str, str] = {}

    def find(group: str) -> str:
        parents.setdefault(group, group)
        if parents[group] != group:
            parents[group] = find(parents[group])
        return parents[group]

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for record in records:
        lesion_group = f"lesion:{record['lesion_id']}"
        find(lesion_group)
        image_hash = record.get("image_hash")
        if image_hash:
            union(lesion_group, f"image:{image_hash}")
    groups = np.asarray([find(f"lesion:{record['lesion_id']}") for record in records])
    classes_per_group: dict[str, set[str]] = {}
    for group, label in zip(groups, labels, strict=True):
        classes_per_group.setdefault(group, set()).add(label)
    inconsistent = [group for group, classes in classes_per_group.items() if len(classes) != 1]
    if inconsistent:
        raise ValueError(
            "A lesion ID maps to multiple diagnosis classes; cannot safely split the dataset."
        )

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_validation, test = next(outer.split(np.zeros(len(labels)), labels, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_relative, validation_relative = next(
        inner.split(
            np.zeros(len(train_validation)),
            labels[train_validation],
            groups[train_validation],
        )
    )
    train = train_validation[train_relative]
    validation = train_validation[validation_relative]

    split_indices = {
        "train": train.tolist(),
        "validation": validation.tolist(),
        "test": test.tolist(),
    }
    group_sets = {
        name: {groups[index] for index in indices}
        for name, indices in split_indices.items()
    }
    if (
        group_sets["train"] & group_sets["validation"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["validation"] & group_sets["test"]
    ):
        raise RuntimeError("Lesion IDs overlap across the training, validation, and test splits.")
    if sorted(index for indices in split_indices.values() for index in indices) != list(
        range(len(records))
    ):
        raise RuntimeError("The dataset split did not include every image exactly once.")
    return split_indices


class HAM10000Dataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        records: list[dict[str, str | Path]],
        indices: list[int],
        class_to_index: dict[str, int],
        transform: Any,
    ):
        self.records = [records[index] for index in indices]
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(Path(record["path"])) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_to_index[str(record["label"])]


def make_transforms() -> tuple[Any, Any]:
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    training = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.1,
                hue=0.02,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return training, evaluation


def classification_metrics(
    actual: list[int], predicted: list[int], class_names: list[str]
) -> dict[str, Any]:
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for target, output in zip(actual, predicted, strict=True):
        matrix[target, output] += 1

    class_metrics: dict[str, dict[str, float]] = {}
    supports = matrix.sum(axis=1)
    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        predicted_positive = int(matrix[:, index].sum())
        support = int(supports[index])
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        class_metrics[name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }

    macro = {
        metric: float(
            np.mean([result[metric] for result in class_metrics.values()])
        )
        for metric in ("precision", "recall", "f1_score")
    }
    total = int(matrix.sum())
    weighted = {
        metric: float(
            sum(result[metric] * result["support"] for result in class_metrics.values())
            / max(total, 1)
        )
        for metric in ("precision", "recall", "f1_score")
    }
    return {
        "accuracy": float(np.trace(matrix) / max(total, 1)),
        "macro_avg": macro,
        "weighted_avg": weighted,
        "per_class": class_metrics,
        "confusion_matrix": matrix.tolist(),
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    actual: list[int] = []
    predicted: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(images)
            loss = loss_function(output, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * labels.size(0)
        actual.extend(labels.detach().cpu().tolist())
        predicted.extend(output.argmax(dim=1).detach().cpu().tolist())
    return total_loss / max(len(loader.dataset), 1), actual, predicted


def _accuracy(actual: list[int], predicted: list[int]) -> float:
    if not actual:
        return 0.0
    return float(np.mean(np.asarray(actual) == np.asarray(predicted)))


def _write_confusion_matrix(
    path: Path, matrix: list[list[int]], class_names: list[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *class_names])
        for class_name, row in zip(class_names, matrix, strict=True):
            writer.writerow([class_name, *row])


def _summarize_splits(
    records: list[dict[str, str | Path]],
    split_indices: dict[str, list[int]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name, indices in split_indices.items():
        result[name] = dict(
            sorted(Counter(str(records[index]["label"]) for index in indices).items())
        )
    return result


def load_split_manifest(
    records: list[dict[str, str | Path]],
    manifest_path: Path,
    seed: int = SEED,
) -> dict[str, list[int]]:
    expected = make_grouped_splits(records, seed)
    record_by_image = {str(record["image_id"]): index for index, record in enumerate(records)}
    manifest_indices: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    seen: set[str] = set()
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["image_id", "lesion_id", "label", "split"]:
            raise ValueError("Split manifest has unexpected columns.")
        for row in reader:
            image_id = row["image_id"]
            if image_id in seen or image_id not in record_by_image:
                raise ValueError(f"Split manifest has an invalid or duplicate image: {image_id}")
            index = record_by_image[image_id]
            record = records[index]
            if row["lesion_id"] != str(record["lesion_id"]) or row["label"] != str(record["label"]):
                raise ValueError(f"Split manifest metadata does not match image {image_id}.")
            if row["split"] not in manifest_indices:
                raise ValueError(f"Split manifest has an invalid split: {row['split']}")
            manifest_indices[row["split"]].append(index)
            seen.add(image_id)
    if seen != set(record_by_image) or manifest_indices != expected:
        raise ValueError("Split manifest does not match the deterministic grouped split.")
    return manifest_indices


def verify_split_manifest(project_root: Path, manifest_path: Path, seed: int = SEED) -> dict[str, int]:
    records = load_records(project_root)
    splits = load_split_manifest(records, manifest_path, seed)
    return {name: len(indices) for name, indices in splits.items()}


def train_and_evaluate(
    project_root: Path,
    output_directory: Path,
    model_directory: Path | None = None,
    epochs: int = 5,
    batch_size: int = 16,
    seed: int = SEED,
    pretrained: bool = True,
    patience: int = 4,
    min_delta: float = 1e-4,
    architecture_names: tuple[str, ...] = MODEL_NAMES,
    weighting: str = "inverse_frequency",
    split_manifest: Path | None = None,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    if epochs < 1 or epochs > 20 or batch_size < 1:
        raise ValueError("Epochs must be between 1 and 20, and batch size must be positive.")
    if patience < 1 or min_delta < 0:
        raise ValueError("Patience must be positive and min_delta cannot be negative.")
    if not architecture_names or any(name not in MODEL_NAMES for name in architecture_names):
        raise ValueError(f"Architectures must be selected from {MODEL_NAMES}.")
    if weighting not in {"inverse_frequency", "mild", "weighted_sampler"}:
        raise ValueError("Unsupported weighting strategy.")
    if model_directory is None:
        raise ValueError(
            "model_directory is required so training cannot overwrite existing model artifacts."
        )
    output_directory = output_directory.resolve()
    model_directory = model_directory.resolve()
    canonical_model_directory = (
        project_root.resolve() / "backend" / "artifacts" / "models"
    )
    if model_directory == canonical_model_directory:
        raise ValueError(
            "Use a separate run-specific model directory; the existing Milestone 2 "
            "model directory is protected."
        )
    for directory, label in (
        (output_directory, "report"),
        (model_directory, "model"),
    ):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"The {label} output directory must be new or empty: {directory}"
            )
        directory.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    records = load_records(project_root)
    class_names = sorted({str(record["label"]) for record in records})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    splits = (
        load_split_manifest(records, split_manifest, seed)
        if split_manifest is not None
        else make_grouped_splits(records, seed)
    )
    with (output_directory / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "lesion_id", "label", "split"])
        split_for_index = {
            index: name for name, indices in splits.items() for index in indices
        }
        for index, record in enumerate(records):
            writer.writerow(
                [
                    record["image_id"],
                    record["lesion_id"],
                    record["label"],
                    split_for_index[index],
                ]
            )

    training_transform, evaluation_transform = make_transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders: dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]] = {}
    train_counts = Counter(str(records[index]["label"]) for index in splits["train"])
    class_weights = torch.tensor(
        [
            len(splits["train"]) / (len(class_names) * train_counts[name])
            for name in class_names
        ],
        dtype=torch.float32,
    )
    class_weights = class_weights / class_weights.mean()
    if weighting == "mild":
        class_weights = class_weights.sqrt()
        class_weights = class_weights / class_weights.mean()
    for split_name in ("train", "validation", "test"):
        if split_name == "test" and not evaluate_test:
            continue
        dataset = HAM10000Dataset(
            records,
            splits[split_name],
            class_to_index,
            training_transform if split_name == "train" else evaluation_transform,
        )
        sampler = None
        shuffle = split_name == "train"
        if split_name == "train" and weighting == "weighted_sampler":
            sample_weights = torch.tensor(
                [1.0 / train_counts[str(record["label"])] for record in dataset.records],
                dtype=torch.double,
            )
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )
            shuffle = False
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    report: dict[str, Any] = {
        "dataset": "HAM10000",
        "image_count": len(records),
        "class_names": class_names,
        "seed": seed,
        "split_method": (
            "StratifiedGroupKFold: 20% held-out test fold, then 25% of the "
            "remaining 80% for validation (approximately 60/20/20); lesion_id "
            "and exact duplicate-image groups never cross splits."
        ),
        "split_counts": {
            name: len(indices) for name, indices in splits.items()
        },
        "class_counts_by_split": _summarize_splits(records, splits),
        "device": str(device),
        "pretrained_imagenet_weights": pretrained,
        "epochs": epochs,
        "batch_size": batch_size,
        "early_stopping_patience": patience,
        "early_stopping_min_delta": min_delta,
        "class_weights": (
            {
                name: float(weight)
                for name, weight in zip(class_names, class_weights.tolist(), strict=True)
            }
            if weighting != "weighted_sampler"
            else None
        ),
        "augmentation": [
            "RandomResizedCrop(scale=0.85..1.0, ratio=0.9..1.1)",
            "RandomHorizontalFlip",
            "RandomVerticalFlip(p=0.15)",
            "RandomRotation(20 degrees)",
            "ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02)",
        ],
        "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=1)",
        "selection_metric": "validation macro F1-score",
        "models": {},
        "weighting": weighting,
        "split_manifest": str(split_manifest.resolve()) if split_manifest else None,
    }

    loss_weights = None if weighting == "weighted_sampler" else class_weights.to(device)
    for architecture in architecture_names:
        model = build_model(architecture, len(class_names), pretrained=pretrained).to(
            device
        )
        loss_function = nn.CrossEntropyLoss(weight=loss_weights)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=2e-4 if architecture == "mobilenet_v2" else 1e-4,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=1,
            threshold=min_delta,
            threshold_mode="abs",
        )
        best_validation_f1 = -1.0
        best_validation_accuracy = -1.0
        best_epoch = 0
        epochs_without_improvement = 0
        checkpoint_path = model_directory / f"{architecture}.pth"
        temporary_checkpoint = model_directory / f".{architecture}.pth.partial"
        validation_history: list[dict[str, float]] = []
        for epoch in range(epochs):
            train_loss, train_actual, train_predicted = _run_epoch(
                model, loaders["train"], loss_function, device, optimizer
            )
            validation_loss, validation_actual, validation_predicted = _run_epoch(
                model, loaders["validation"], loss_function, device, None
            )
            validation_metrics = classification_metrics(
                validation_actual, validation_predicted, class_names
            )
            validation_f1 = validation_metrics["macro_avg"]["f1_score"]
            validation_accuracy = validation_metrics["accuracy"]
            scheduler.step(validation_f1)
            validation_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": _accuracy(train_actual, train_predicted),
                    "train_macro_f1": classification_metrics(
                        train_actual, train_predicted, class_names
                    )["macro_avg"]["f1_score"],
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "validation_macro_precision": validation_metrics["macro_avg"][
                        "precision"
                    ],
                    "validation_macro_recall": validation_metrics["macro_avg"]["recall"],
                    "validation_macro_f1": validation_f1,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            print(
                f"{architecture} epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f}, validation_loss={validation_loss:.4f}, "
                f"train_accuracy={validation_history[-1]['train_accuracy']:.4f}, "
                f"validation_accuracy={validation_accuracy:.4f}, "
                f"validation_macro_f1={validation_f1:.4f}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
            improved = (
                validation_f1 > best_validation_f1 + min_delta
                or (
                    abs(validation_f1 - best_validation_f1) <= min_delta
                    and validation_accuracy > best_validation_accuracy
                )
            )
            if improved:
                best_validation_f1 = validation_f1
                best_validation_accuracy = validation_accuracy
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                # Never replace a usable published model while a run is in progress.
                torch.save(model.state_dict(), temporary_checkpoint)
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"{architecture} early stopping at epoch {epoch + 1}; "
                    f"best epoch was {best_epoch}.",
                    flush=True,
                )
                break

        model.load_state_dict(
            torch.load(temporary_checkpoint, map_location=device, weights_only=True)
        )
        validation_loss, validation_actual, validation_predicted = _run_epoch(
            model, loaders["validation"], loss_function, device, None
        )
        validation_metrics = classification_metrics(
            validation_actual, validation_predicted, class_names
        )
        os.replace(temporary_checkpoint, checkpoint_path)
        _write_confusion_matrix(
            output_directory / f"{architecture}_validation_confusion_matrix.csv",
            validation_metrics["confusion_matrix"],
            class_names,
        )
        report["models"][architecture] = {
            "weights": checkpoint_path.name,
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
            "test_loss": None,
            "test_metrics": None,
            "history": validation_history,
            "best_epoch": best_epoch,
            "stopped_epoch": len(validation_history),
            "early_stopped": len(validation_history) < epochs,
            "overfitting_gap_at_best_epoch": next(
                (
                    entry["train_accuracy"] - entry["validation_accuracy"]
                    for entry in validation_history
                    if entry["epoch"] == best_epoch
                ),
                None,
            ),
            "hyperparameters": {
                "learning_rate": 2e-4 if architecture == "mobilenet_v2" else 1e-4,
                "weight_decay": 1e-4,
                "optimizer": "AdamW",
            },
        }
        print(
            f"{architecture} final: validation_macro_f1="
            f"{validation_metrics['macro_avg']['f1_score']:.4f}",
            flush=True,
        )

    best_architecture = max(
        architecture_names,
        key=lambda name: (
            report["models"][name]["validation_metrics"]["macro_avg"]["f1_score"],
            report["models"][name]["validation_metrics"]["accuracy"],
        ),
    )
    selected_model = build_model(best_architecture, len(class_names)).to(device)
    selected_model.load_state_dict(
        torch.load(
            model_directory / report["models"][best_architecture]["weights"],
            map_location=device,
            weights_only=True,
        )
    )
    if evaluate_test:
        selected_test_loss, test_actual, test_predicted = _run_epoch(
            selected_model,
            loaders["test"],
            nn.CrossEntropyLoss(weight=loss_weights),
            device,
            None,
        )
        selected_test_metrics = classification_metrics(
            test_actual, test_predicted, class_names
        )
        report["models"][best_architecture]["test_loss"] = selected_test_loss
        report["models"][best_architecture]["test_metrics"] = selected_test_metrics
        _write_confusion_matrix(
            output_directory / f"{best_architecture}_test_confusion_matrix.csv",
            selected_test_metrics["confusion_matrix"],
            class_names,
        )
    report["selected_model"] = best_architecture
    (model_directory / "selection.json").write_text(
        json.dumps(
            {
                "architecture": best_architecture,
                "weights": report["models"][best_architecture]["weights"],
                "class_names": class_names,
                "selection_metric": "validation macro F1-score",
                "weighting": weighting,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_directory / "evaluation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    report_path = output_directory / "evaluation_report.md"
    lines = [
        "# DermAI Milestone 2 model evaluation",
        "",
        f"- Dataset: HAM10000, {len(records)} real dermoscopic images.",
        f"- Classes: {', '.join(class_names)}.",
        f"- Device: {device}.",
        f"- Training: ImageNet-pretrained weights; final feature stage and classifier head fine-tuned for up to {epochs} epoch(s) per model with batch size {batch_size}.",
        f"- Early stopping: patience={patience}, min_delta={min_delta}; ReduceLROnPlateau halves the learning rate after one validation macro-F1 plateau.",
        f"- Split: {report['split_method']}",
        f"- Split sizes: {report['split_counts']}.",
        f"- Model selection: highest validation macro F1-score; test metrics were not used to select the model.",
        f"- Selected model: **{best_architecture}**.",
        "",
        "| Model | Best epoch | Stopped epoch | Val accuracy | Val macro precision | Val macro recall | Val macro F1 | Test accuracy | Test macro precision | Test macro recall | Test macro F1 | Overfit gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for architecture in architecture_names:
        result = report["models"][architecture]
        validation = result["validation_metrics"]
        test = result["test_metrics"]
        test_values = (
            f"{test['accuracy']:.4f} | {test['macro_avg']['precision']:.4f} | "
            f"{test['macro_avg']['recall']:.4f} | {test['macro_avg']['f1_score']:.4f}"
            if test is not None
            else "— | — | — | —"
        )
        lines.append(
            f"| {architecture} | {result['best_epoch']} | {result['stopped_epoch']} | "
            f"{validation['accuracy']:.4f} | "
            f"{validation['macro_avg']['precision']:.4f} | "
            f"{validation['macro_avg']['recall']:.4f} | "
            f"{validation['macro_avg']['f1_score']:.4f} | {test_values} | "
            f"{result['overfitting_gap_at_best_epoch']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Confusion matrices",
            "",
            "Validation confusion matrices for both models and a held-out test confusion matrix for the selected model are saved as CSV files alongside this report. Full per-class precision, recall, F1-score, and support are in `evaluation_report.json`.",
            "",
            "## Lesion detection status",
            "",
            "Pending. HAM10000 provides diagnosis labels and lesion IDs, but no pixel-level masks or bounding boxes. Classification outputs are not lesion detections.",
            "",
            "## Limitations",
            "",
            "HAM10000 contains dermoscopic images, not ordinary mobile-phone photographs. The predicted class and confidence are research outputs and are not a diagnosis.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def evaluate_checkpoint(
    project_root: Path,
    checkpoint_directory: Path,
    output_directory: Path,
    split_manifest: Path,
    seed: int = SEED,
) -> dict[str, Any]:
    output_directory = output_directory.resolve()
    checkpoint_directory = checkpoint_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Evaluation output directory must be new or empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    records = load_records(project_root)
    splits = load_split_manifest(records, split_manifest.resolve(), seed)
    selection_path = checkpoint_directory / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Checkpoint selection metadata was not found: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    architecture = selection["architecture"]
    class_names = selection["class_names"]
    checkpoint_path = checkpoint_directory / selection["weights"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected checkpoint was not found: {checkpoint_path}")
    _, evaluation_transform = make_transforms()
    dataset = HAM10000Dataset(
        records, splits["test"], {name: index for index, name in enumerate(class_names)}, evaluation_transform
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(architecture, len(class_names)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_loss, actual, predicted = _run_epoch(model, loader, nn.CrossEntropyLoss(), device, None)
    metrics = classification_metrics(actual, predicted, class_names)
    _write_confusion_matrix(
        output_directory / f"{architecture}_test_confusion_matrix.csv",
        metrics["confusion_matrix"],
        class_names,
    )
    report = {
        "architecture": architecture,
        "checkpoint": str(checkpoint_path),
        "split_manifest": str(split_manifest.resolve()),
        "test_loss": test_loss,
        "test_metrics": metrics,
        "selection_metric": selection["selection_metric"],
    }
    (output_directory / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
