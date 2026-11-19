"""
DermAI Milestone 2: General Model Training Script.

Features:
- Supported architectures: resnet50, efficientnet_b0, densenet121, mobilenetv3_large_100, etc. (via timm).
- Two-stage transfer learning:
    Stage 1: Frozen backbone, trains classification head for --freeze_epochs.
    Stage 2: Full network fine-tuning with differential learning rates.
- Loss functions:
    'ce_weighted': Inverse frequency class-weighted cross-entropy (computed from the FULL training split).
    'focal': Multi-class focal loss with alpha weighting and gamma focusing parameter.
- Stratified random sampling for --subset N (guarantees all 7 classes appear, minimum 2 per class).
- AdamW optimizer with CosineAnnealingLR or ReduceLROnPlateau scheduler.
- Early stopping monitored on validation Macro-F1.
- Automatic Mixed Precision (AMP) on GPU only.
- Validation artifacts saved at end of every run:
    best.pt, last.pt, history.csv, training_curves.png, val_confusion_matrix.png,
    val_classification_report.txt, and metrics.json.
- Opt-in test evaluation via --eval_test (default False), printing strict final-evaluation warning.
- Full --resume capability for Kaggle timeouts.
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

from ml.training.dataset import (
    get_dataloaders,
    compute_class_weights,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    save_classes_json
)


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# Loss Functions
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples and focuses on hard/minority skin lesion samples.
    """
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_criterion(
    loss_type: str,
    class_weights: torch.Tensor,
    device: torch.device
) -> nn.Module:
    """Instantiates the specified loss function."""
    weights_dev = class_weights.to(device)
    if loss_type == "ce_weighted":
        return nn.CrossEntropyLoss(weight=weights_dev)
    elif loss_type == "focal":
        return FocalLoss(alpha=weights_dev, gamma=2.0)
    else:
        raise ValueError(f"Unknown loss type: '{loss_type}'. Expected 'ce_weighted' or 'focal'.")


# ==============================================================================
# Model Factory & Transfer Learning Helpers
# ==============================================================================

def create_classifier_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True
) -> nn.Module:
    """
    Creates a model via timm with ImageNet pretrained weights, replacing
    the final classification layer with num_classes (7) outputs.
    """
    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        return model
    except Exception as e:
        raise ValueError(f"Failed to create model '{model_name}' via timm: {e}")


def freeze_backbone(model: nn.Module) -> None:
    """Freezes all backbone layers and leaves only the classification head trainable."""
    for param in model.parameters():
        param.requires_grad = False

    classifier = model.get_classifier()
    for param in classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreezes all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ==============================================================================
# Evaluation Routine
# ==============================================================================

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluates the model on a given data loader.
    Returns:
        (loss, accuracy, macro_f1, balanced_accuracy, y_true, y_pred)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = labels.size(0)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    avg_loss = total_loss / max(1, total_samples)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    return avg_loss, acc, macro_f1, balanced_acc, y_true, y_pred


# ==============================================================================
# Plotting Routines
# ==============================================================================

def save_training_curves(history: List[Dict[str, Any]], output_path: Path) -> None:
    """Generates and saves side-by-side training progression curves."""
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_macro_f1 = [h["val_macro_f1"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_balanced_acc = [h["val_balanced_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curves
    ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    ax1.set_title("Loss Progression", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Metric Curves
    ax2.plot(epochs, val_macro_f1, label="Val Macro-F1 (Early Stop Target)", color="#2ca02c", linewidth=2.5)
    ax2.plot(epochs, val_balanced_acc, label="Val Balanced Acc", color="#9467bd", linewidth=1.8, linestyle="--")
    ax2.plot(epochs, val_acc, label="Val Accuracy", color="#7f7f7f", linewidth=1.5, linestyle=":")
    ax2.set_title("Validation Classification Metrics", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Validation Confusion Matrix"
) -> None:
    """Generates and saves an annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Diagnosis",
        xlabel="Predicted Diagnosis"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9
            )
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ==============================================================================
# Main Training Loop
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DermAI Model Training Pipeline")
    # Architecture and Hyperparameters
    parser.add_argument("--model", type=str, default="resnet50",
                        help="Model architecture name in timm (e.g. resnet50, efficientnet_b0, densenet121, mobilenetv3_large_100)")
    parser.add_argument("--epochs", type=int, default=25, help="Total training epochs (default: 25)")
    parser.add_argument("--freeze_epochs", type=int, default=3,
                        help="Epochs to train only the classification head before unfreezing backbone (default: 3)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation (default: 32)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for head / initial training (default: 3e-4)")
    parser.add_argument("--lr_fine_tune", type=float, default=3e-5, help="Learning rate for fine-tuning unfrozen backbone (default: 3e-5)")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW optimizer (default: 1e-2)")
    parser.add_argument("--loss", type=str, default="ce_weighted", choices=["ce_weighted", "focal"],
                        help="Loss function: 'ce_weighted' or 'focal' (default: ce_weighted)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau"],
                        help="Learning rate scheduler (default: cosine)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience in epochs based on val macro-F1 (default: 7)")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum improvement in macro-F1 to reset patience (default: 1e-4)")

    # Data and Paths
    parser.add_argument("--splits_csv", type=str, default="ml/data/splits.csv", help="Path to splits.csv")
    parser.add_argument("--images_root", type=str, nargs="+", default=["ml/data/raw/images"],
                        help="One or more folders containing lesion images (e.g. for Kaggle Part 1 & 2)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ml/experiments/<model>)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Target images per split for stratified random smoke tests (guarantees all 7 classes appear)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default: 0 for Windows)")
    parser.add_argument("--use_sampler", action="store_true", default=False,
                        help="Use WeightedRandomSampler for training split")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Compute device: auto, cuda, or cpu (default: auto)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume training from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no_pretrained", action="store_true", default=False, help="Do not load ImageNet weights (scratch training)")
    parser.add_argument("--eval_test", action="store_true", default=False,
                        help="Opt-in flag to run final evaluation on the test split. Use ONLY for the final selected model.")
    return parser.parse_args()


def train():
    args = parse_args()
    set_seed(args.seed)

    # 1. Resolve Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    use_amp = (device.type == "cuda")

    # 2. Setup Output Directory
    if args.output_dir is None:
        output_dir = Path(f"ml/experiments/{args.model}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save classes.json
    save_classes_json(output_dir / "classes.json")

    # Save experiment configuration
    config_dict = vars(args).copy()
    config_dict["device_actual"] = str(device)
    config_dict["amp_enabled"] = use_amp
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    print("=" * 75)
    print(f"DermAI Training: Model '{args.model}'")
    print("=" * 75)
    print(f"Device               : {device} (AMP: {use_amp})")
    print(f"Epochs               : {args.epochs} (Head freeze: {args.freeze_epochs} epochs)")
    print(f"Batch Size           : {args.batch_size}")
    print(f"Learning Rate        : {args.lr} (Fine-tune: {args.lr_fine_tune})")
    print(f"Loss Function        : {args.loss}")
    print(f"Optimizer / Sched    : AdamW / {args.scheduler}")
    print(f"Early Stop Patience  : {args.patience} epochs (monitored on val macro-F1)")
    print(f"Output Directory     : {output_dir}")
    print(f"Subset per Split     : {args.subset if args.subset else 'Full Dataset (No Subset)'}")
    print(f"Eval Test Split      : {args.eval_test}")
    print("-" * 75)

    # 3. DataLoaders & Class Weights (From FULL training split only)
    loaders = get_dataloaders(
        splits_csv=args.splits_csv,
        images_root=args.images_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset=args.subset,
        use_sampler=args.use_sampler,
        seed=args.seed
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # Print class counts of each split/subset at startup to verify all 7 classes appear
    train_counts = train_loader.dataset.get_class_counts()
    val_counts = val_loader.dataset.get_class_counts()
    test_counts = test_loader.dataset.get_class_counts()

    print("\n--- Per-Class Sample Counts at Startup ---")
    print(f"  {'Class':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("  " + "-" * 32)
    for dx in CLASS_TO_IDX.keys():
        print(f"  {dx:<8} {train_counts[dx]:<8} {val_counts[dx]:<8} {test_counts[dx]:<8}")
    print("  " + "-" * 32)
    print(f"  {'Total':<8} {len(train_loader.dataset):<8} {len(val_loader.dataset):<8} {len(test_loader.dataset):<8}")

    # Compute class weights strictly from the FULL training split
    class_weights = compute_class_weights(args.splits_csv)
    print("\n--- Full Training Split Inverse-Frequency Weights ---")
    for dx, idx in sorted(CLASS_TO_IDX.items(), key=lambda x: x[1]):
        print(f"  [{idx}] {dx:<6} : weight = {class_weights[idx].item():8.4f}")

    criterion = build_criterion(args.loss, class_weights, device)

    # 4. Model Initialization
    pretrained = not args.no_pretrained
    model = create_classifier_model(args.model, num_classes=NUM_CLASSES, pretrained=pretrained)
    model = model.to(device)

    # 5. Optimizer & AMP Scaler
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_macro_f1 = -1.0
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    # 6. Resume Handling
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"\n[INFO] Loading resume checkpoint from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_macro_f1 = ckpt.get("best_val_macro_f1", -1.0)
        patience_counter = ckpt.get("patience_counter", 0)

        hist_file = output_dir / "history.csv"
        if hist_file.is_file():
            history = pd.read_csv(hist_file).to_dict(orient="records")
        print(f"[INFO] Resumed successfully. Starting from epoch {start_epoch} (Best Macro-F1 so far: {best_macro_f1:.4f})")

    # Helper function to configure optimizer for the active stage
    def setup_optimizer_and_scheduler(current_epoch: int):
        is_frozen = current_epoch <= args.freeze_epochs
        if is_frozen:
            print(f"\n>>> Stage 1: Frozen Backbone (Training Head Only @ lr={args.lr:.2e}) <<<")
            freeze_backbone(model)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
            stage_epochs = max(1, args.freeze_epochs)
        else:
            print(f"\n>>> Stage 2: Fine-Tuning All Layers (Backbone lr={args.lr_fine_tune:.2e}, Head lr={args.lr:.2e}) <<<")
            unfreeze_all(model)
            classifier_params = set(model.get_classifier().parameters())
            backbone_params = [p for p in model.parameters() if p not in classifier_params]
            opt = torch.optim.AdamW([
                {"params": backbone_params, "lr": args.lr_fine_tune},
                {"params": list(classifier_params), "lr": args.lr}
            ], weight_decay=args.weight_decay)
            stage_epochs = max(1, args.epochs - args.freeze_epochs)

        if args.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=stage_epochs, eta_min=1e-6)
        elif args.scheduler == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, min_lr=1e-6)
        else:
            sched = None

        return opt, sched

    current_stage_is_frozen = None
    optimizer = None
    scheduler = None

    # 7. Training Loop
    print("\nStarting Training Execution...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        stage_is_frozen = (epoch <= args.freeze_epochs)

        if stage_is_frozen != current_stage_is_frozen or optimizer is None:
            optimizer, scheduler = setup_optimizer_and_scheduler(epoch)
            current_stage_is_frozen = stage_is_frozen

        model.train()
        running_loss = 0.0
        train_samples = 0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_sz = labels.size(0)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * batch_sz
            train_samples += batch_sz

        train_loss = running_loss / max(1, train_samples)

        # Validation phase
        val_loss, val_acc, val_macro_f1, val_balanced_acc, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_macro_f1)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_duration = time.time() - epoch_start
        stage_tag = "Head" if stage_is_frozen else "Full"

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({stage_tag}) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Val Bal-Acc: {val_balanced_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_duration:.1f}s"
        )

        epoch_record = {
            "epoch": epoch,
            "stage": stage_tag,
            "lr": current_lr,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "val_macro_f1": round(val_macro_f1, 5),
            "val_balanced_acc": round(val_balanced_acc, 5),
            "time_sec": round(epoch_duration, 2)
        }
        history.append(epoch_record)

        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        save_training_curves(history, output_dir / "training_curves.png")

        # Save last.pt
        last_ckpt = {
            "epoch": epoch,
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "best_val_macro_f1": best_macro_f1,
            "patience_counter": patience_counter,
            "config": vars(args)
        }
        torch.save(last_ckpt, output_dir / "last.pt")

        # Best model checkpoint check
        if val_macro_f1 > best_macro_f1 + args.min_delta:
            improvement = val_macro_f1 - best_macro_f1
            best_macro_f1 = val_macro_f1
            patience_counter = 0

            best_ckpt = {
                "epoch": epoch,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "best_val_macro_f1": best_macro_f1,
                "val_acc": val_acc,
                "val_balanced_acc": val_balanced_acc,
                "val_loss": val_loss,
                "config": vars(args),
                "classes": CLASS_TO_IDX
            }
            torch.save(best_ckpt, output_dir / "best.pt")
            print(f"  --> Saved new BEST checkpoint (Val Macro-F1: {best_macro_f1:.4f}, +{improvement:.4f})")
        else:
            patience_counter += 1
            print(f"  --> Patience counter: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"\n[EARLY STOPPING TRIGGERED] No improvement in val macro-F1 for {args.patience} consecutive epochs.")
            break

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time / 60:.2f} minutes.")
    print(f"Best Validation Macro-F1 achieved: {best_macro_f1:.4f}")

    # ==============================================================================
    # 8. Post-Training Validation Evaluation (Standard at end of every run)
    # ==============================================================================
    print("\n" + "=" * 75)
    print("FINAL VALIDATION EVALUATION (Using best.pt)")
    print("=" * 75)
    best_pt_path = output_dir / "best.pt"
    if best_pt_path.is_file():
        ckpt = torch.load(best_pt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch']} (Best Val Macro-F1: {ckpt['best_val_macro_f1']:.4f})")

    val_loss, val_acc, val_macro_f1, val_bal_acc, y_true_val, y_pred_val = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp
    )

    print(f"Final Val Loss         : {val_loss:.4f}")
    print(f"Final Val Accuracy     : {val_acc:.4f}")
    print(f"Final Val Macro-F1     : {val_macro_f1:.4f}")
    print(f"Final Val Balanced Acc : {val_bal_acc:.4f}")

    target_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    all_class_labels = list(range(NUM_CLASSES))

    val_clf_report_dict = classification_report(
        y_true_val,
        y_pred_val,
        labels=all_class_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    val_clf_report_text = classification_report(
        y_true_val,
        y_pred_val,
        labels=all_class_labels,
        target_names=target_names,
        zero_division=0
    )

    print("\nValidation Classification Report:")
    print(val_clf_report_text)

    # Save validation classification report and confusion matrix
    with open(output_dir / "val_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(val_clf_report_text)

    save_confusion_matrix(
        y_true_val,
        y_pred_val,
        class_names=target_names,
        output_path=output_dir / "val_confusion_matrix.png",
        title=f"Validation Confusion Matrix - {args.model}"
    )

    metrics_payload = {
        "model": args.model,
        "best_epoch": ckpt.get("epoch") if best_pt_path.is_file() else None,
        "val": {
            "loss": round(val_loss, 5),
            "accuracy": round(val_acc, 5),
            "macro_f1": round(val_macro_f1, 5),
            "balanced_accuracy": round(val_bal_acc, 5),
            "classification_report": val_clf_report_dict
        }
    }

    # ==============================================================================
    # 9. Opt-in Final Test Split Evaluation (--eval_test only)
    # ==============================================================================
    if args.eval_test:
        print("\n" + "=" * 75)
        print("WARNING: Test split must only be used once, for the final chosen model.")
        print("=" * 75)

        test_loss, test_acc, test_macro_f1, test_bal_acc, y_true_test, y_pred_test = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp
        )

        print(f"Test Loss            : {test_loss:.4f}")
        print(f"Test Accuracy        : {test_acc:.4f}")
        print(f"Test Macro-F1        : {test_macro_f1:.4f}")
        print(f"Test Balanced Acc    : {test_bal_acc:.4f}")

        test_clf_report_dict = classification_report(
            y_true_test,
            y_pred_test,
            labels=all_class_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        test_clf_report_text = classification_report(
            y_true_test,
            y_pred_test,
            labels=all_class_labels,
            target_names=target_names,
            zero_division=0
        )

        print("\nTest Classification Report:")
        print(test_clf_report_text)

        with open(output_dir / "test_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(test_clf_report_text)

        save_confusion_matrix(
            y_true_test,
            y_pred_test,
            class_names=target_names,
            output_path=output_dir / "test_confusion_matrix.png",
            title=f"Test Confusion Matrix - {args.model}"
        )

        metrics_payload["test"] = {
            "loss": round(test_loss, 5),
            "accuracy": round(test_acc, 5),
            "macro_f1": round(test_macro_f1, 5),
            "balanced_accuracy": round(test_bal_acc, 5),
            "classification_report": test_clf_report_dict
        }
    else:
        print("\n[NOTE] Test split evaluation skipped (--eval_test not set). Test split preserved for final evaluation.")

    # Save metrics JSON
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\n[OK] Run artifacts successfully saved in {output_dir}:")
    print(f"  - {output_dir / 'best.pt'}")
    print(f"  - {output_dir / 'last.pt'}")
    print(f"  - {output_dir / 'history.csv'}")
    print(f"  - {output_dir / 'training_curves.png'}")
    print(f"  - {output_dir / 'val_confusion_matrix.png'}")
    print(f"  - {output_dir / 'val_classification_report.txt'}")
    print(f"  - {output_dir / 'metrics.json'}")
    if args.eval_test:
        print(f"  - {output_dir / 'test_confusion_matrix.png'}")
        print(f"  - {output_dir / 'test_classification_report.txt'}")


if __name__ == "__main__":
    train()
