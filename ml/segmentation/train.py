"""
Training script for ISIC 2018 Task 1 Skin Lesion Segmentation.

Features:
- U-Net with ResNet34 pretrained encoder (segmentation_models_pytorch).
- Joint Dice + Binary Cross-Entropy Loss.
- Metrics per epoch: Dice coefficient, IoU, Pixel Accuracy, Sensitivity, Specificity.
- Optimizer: AdamW with CosineAnnealingLR / ReduceLROnPlateau.
- Tracks best model by validation Dice -> saves to `best_model.pth`.
- Saves training curves (loss, Dice, IoU, sensitivity, specificity) to `training_curves.png`.
- Produces a figure of 8 test examples displaying Image, Ground Truth Mask, Predicted Mask, and Overlay.
- Saves training metrics history to `history.csv` and training parameters to `config.json`.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.segmentation.dataset import (
    discover_image_mask_pairs,
    split_dataset,
    create_smoke_dataset,
    get_segmentation_transforms,
    ISICSegmentationDataset,
)
from ml.segmentation.model import build_segmentation_unet
from ml.segmentation.losses import DiceBCELoss
from ml.segmentation.metrics import SegmentationMeter, calculate_segmentation_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.train()
    meter = SegmentationMeter()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        meter.update(logits.detach(), masks.detach(), loss=loss.item())

    return meter.compute()


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    meter = SegmentationMeter()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        meter.update(logits, masks, loss=loss.item())

    return meter.compute()


def plot_training_curves(history: List[Dict[str, Any]], save_path: Path) -> None:
    """Plots and saves loss, Dice, IoU, Sensitivity, Specificity, and Pixel Accuracy curves."""
    df = pd.DataFrame(history)
    epochs = df["epoch"].values

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Loss
    axes[0, 0].plot(epochs, df["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    axes[0, 0].plot(epochs, df["val_loss"], "s-", label="Val Loss", color="#ff7f0e")
    axes[0, 0].set_title("Loss (Dice + BCE)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Dice Coefficient
    axes[0, 1].plot(epochs, df["train_dice"], "o-", label="Train Dice", color="#1f77b4")
    axes[0, 1].plot(epochs, df["val_dice"], "s-", label="Val Dice", color="#2ca02c")
    axes[0, 1].set_title("Dice Coefficient")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. IoU (Jaccard Index)
    axes[0, 2].plot(epochs, df["train_iou"], "o-", label="Train IoU", color="#1f77b4")
    axes[0, 2].plot(epochs, df["val_iou"], "s-", label="Val IoU", color="#9467bd")
    axes[0, 2].set_title("IoU (Jaccard Index)")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    # 4. Sensitivity (Recall)
    axes[1, 0].plot(epochs, df["val_sensitivity"], "s-", label="Val Sensitivity", color="#d62728")
    axes[1, 0].set_title("Validation Sensitivity (Recall)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 5. Specificity
    axes[1, 1].plot(epochs, df["val_specificity"], "s-", label="Val Specificity", color="#8c564b")
    axes[1, 1].set_title("Validation Specificity")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    # 6. Pixel Accuracy
    axes[1, 2].plot(epochs, df["val_pixel_accuracy"], "s-", label="Val Pixel Accuracy", color="#e377c2")
    axes[1, 2].set_title("Validation Pixel Accuracy")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def save_test_examples_figure(
    model: nn.Module,
    samples: List[Dict[str, str]],
    save_path: Path,
    device: torch.device,
    num_examples: int = 8,
    img_size: int = 256,
) -> None:
    """
    Saves a figure of test examples showing:
    Original Image | Ground Truth Mask | Predicted Mask | Overlay
    """
    model.eval()
    if not samples:
        return

    n = min(num_examples, len(samples))
    chosen = samples[:n]

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    col_headers = ["Original Image", "Ground Truth Mask", "Predicted Mask", "Prediction Overlay"]

    for i, item in enumerate(chosen):
        img_id = item["image_id"]
        img_bgr = cv2.imread(item["image_path"])
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        # Ground truth mask
        mask_raw = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            continue
        mask_true = cv2.resize((mask_raw > 127).astype(np.uint8) * 255, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        # Model inference
        norm = (img_resized.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        mask_pred = (probs >= 0.5).astype(np.uint8) * 255

        # Compute sample Dice
        sample_metrics = calculate_segmentation_metrics(mask_pred / 255.0, mask_true / 255.0)
        dice_score = sample_metrics["dice"]

        # Overlay
        overlay = img_resized.copy()
        color_mask = np.zeros_like(img_resized)
        color_mask[mask_pred > 0] = [0, 255, 0]  # Green overlay
        cv2.addWeighted(color_mask, 0.45, overlay, 0.55, 0, overlay)
        contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # Plot 4 columns
        axes[i, 0].imshow(img_resized)
        axes[i, 0].set_title(f"{img_id}\n(Image)", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_true, cmap="gray")
        axes[i, 1].set_title("Ground Truth", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(mask_pred, cmap="gray")
        axes[i, 2].set_title(f"Predicted\n(Dice={dice_score:.3f})", fontsize=10)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Overlay", fontsize=10)
        axes[i, 3].axis("off")

    for col_idx, header in enumerate(col_headers):
        axes[0, col_idx].set_title(f"{header}", fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_training(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check data availability
    pairs = discover_image_mask_pairs(data_dir)
    if not pairs:
        print(f"[WARN] No image/mask pairs found in {data_dir}.")
        print("[INFO] Generating synthetic smoke sample dataset for testing...")
        create_smoke_dataset(output_dir=data_dir, num_samples=max(24, args.subset or 24), seed=args.seed)
        pairs = discover_image_mask_pairs(data_dir)
        print(f"[INFO] Generated {len(pairs)} smoke testing image/mask pairs.")

    print(f"[INFO] Total discovered image/mask pairs: {len(pairs)}")

    # Subset if requested
    if args.subset and args.subset < len(pairs):
        rng = np.random.RandomState(args.seed)
        sampled_indices = rng.choice(len(pairs), args.subset, replace=False)
        pairs = [pairs[i] for i in sampled_indices]
        print(f"[INFO] Using subset of {len(pairs)} image/mask pairs.")

    # 2. Split dataset 80 / 10 / 10
    splits_csv_path = output_dir / "splits_segmentation.csv"
    split_dict = split_dataset(pairs, ratios=(0.8, 0.1, 0.1), seed=args.seed, save_csv_path=splits_csv_path)
    train_samples = split_dict["train"]
    val_samples = split_dict["val"]
    test_samples = split_dict["test"]

    # In case subsets are very small (e.g. smoke test with 5 samples)
    if not val_samples:
        val_samples = train_samples
    if not test_samples:
        test_samples = val_samples

    print(f"[INFO] Split sizes -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    print(f"[INFO] Saved segmentation splits to: {splits_csv_path}")

    # 3. Create Datasets and DataLoaders
    train_transform = get_segmentation_transforms("train", img_size=args.img_size)
    val_transform = get_segmentation_transforms("val", img_size=args.img_size)

    train_dataset = ISICSegmentationDataset(train_samples, transform=train_transform, img_size=args.img_size)
    val_dataset = ISICSegmentationDataset(val_samples, transform=val_transform, img_size=args.img_size)
    test_dataset = ISICSegmentationDataset(test_samples, transform=val_transform, img_size=args.img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 4. Device and Model
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = (device.type == "cuda")
    print(f"[INFO] Using device: {device} (AMP: {use_amp})")

    model = build_segmentation_unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    # 5. Loss, Optimizer, Scheduler
    criterion = DiceBCELoss(dice_weight=args.dice_weight, bce_weight=args.bce_weight)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    # Save configuration
    config = {
        "model": "unet_resnet34",
        "encoder_weights": "imagenet",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "img_size": args.img_size,
        "seed": args.seed,
        "dice_weight": args.dice_weight,
        "bce_weight": args.bce_weight,
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_test": len(test_samples),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    best_val_dice = -1.0
    best_epoch = 0
    history: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("  STARTING SEGMENTATION TRAINING")
    print("=" * 70)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp=use_amp)
        val_metrics = evaluate_split(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()
        epoch_sec = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_sec": round(epoch_sec, 2),
        }
        history.append(row)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
            f"Dice: {train_metrics['dice']:.4f}/{val_metrics['dice']:.4f} | "
            f"IoU: {train_metrics['iou']:.4f}/{val_metrics['iou']:.4f} | "
            f"Sens: {val_metrics['sensitivity']:.4f} | "
            f"Spec: {val_metrics['specificity']:.4f} | "
            f"PixAcc: {val_metrics['pixel_accuracy']:.4f} | "
            f"Time: {epoch_sec:.1f}s"
        )

        # Checkpoint if best validation Dice
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "config": config,
            }, output_dir / "best_model.pth")

        # Always save last model
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_val_dice": best_val_dice,
        }, output_dir / "last_model.pth")

    total_time_min = (time.time() - start_time) / 60.0
    print("=" * 70)
    print(f"[DONE] Training completed in {total_time_min:.2f} mins.")
    print(f"[INFO] Best Validation Dice: {best_val_dice:.4f} (Epoch {best_epoch})")
    print(f"[SAVED] Best checkpoint -> {output_dir / 'best_model.pth'}")

    # 6. Save history CSV
    history_df = pd.DataFrame(history)
    history_csv = output_dir / "history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"[SAVED] History metrics -> {history_csv}")

    # 7. Plot training curves
    curves_path = output_dir / "training_curves.png"
    plot_training_curves(history, curves_path)
    print(f"[SAVED] Training curves -> {curves_path}")

    # 8. Load best model and evaluate on test split
    best_ckpt = torch.load(output_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate_split(model, test_loader, criterion, device, use_amp=use_amp)

    print("\n" + "=" * 70)
    print("  TEST SPLIT FINAL EVALUATION (Best Model)")
    print(f"  Dice Coefficient : {test_metrics['dice']:.4f}")
    print(f"  IoU (Jaccard)    : {test_metrics['iou']:.4f}")
    print(f"  Pixel Accuracy   : {test_metrics['pixel_accuracy']:.4f}")
    print(f"  Sensitivity      : {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity      : {test_metrics['specificity']:.4f}")
    print("=" * 70)

    with open(output_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # 9. Save 8 test examples figure
    test_fig_path = output_dir / "test_predictions_8examples.png"
    examples_pool = test_samples if len(test_samples) >= 8 else (test_samples + val_samples)
    save_test_examples_figure(
        model=model,
        samples=examples_pool,
        save_path=test_fig_path,
        device=device,
        num_examples=8,
        img_size=args.img_size,
    )
    print(f"[SAVED] 8 Test Examples Figure -> {test_fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net with ResNet34 encoder for Lesion Segmentation.")
    parser.add_argument("--data_dir", type=str, default="ml/data/isic2018_task1", help="Path to ISIC 2018 Task 1 data.")
    parser.add_argument("--output_dir", type=str, default="ml/experiments/segmentation", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--img_size", type=int, default=256, help="Image resize dimension.")
    parser.add_argument("--dice_weight", type=float, default=1.0, help="Weight for Dice loss.")
    parser.add_argument("--bce_weight", type=float, default=1.0, help="Weight for BCE loss.")
    parser.add_argument("--subset", type=int, default=None, help="Optional subset size for fast training or smoke test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu).")
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()

