"""
DermAI Milestone 2: Standalone Model Evaluation Script.

Loads a trained model checkpoint (best.pt) and evaluates it on the val or test split.

Metrics computed:
  - Accuracy, Balanced Accuracy
  - Macro and Weighted Precision, Recall, F1
  - Per-class Precision, Recall, F1 (from sklearn classification_report)
  - Per-class Sensitivity (= Recall) and Specificity (TN / (TN + FP))
  - Per-class one-vs-rest AUC-ROC, and Macro AUC-ROC
  - Matthews Correlation Coefficient (MCC)
  - Per-class TP, FP, TN, FN counts

Plots saved:
  - Confusion matrix (raw counts)
  - Confusion matrix (row-normalized, as % of true class)
  - Per-class ROC curves (one-vs-rest) with AUC annotated

RULES:
  - Val split: used freely for analysis and model comparison.
  - Test split: ONLY for the single final chosen model. A warning is printed.

Usage (val):
    python -m ml.evaluation.evaluate \\
        --checkpoint ml/experiments/resnet50/best.pt \\
        --split val \\
        --images_root ml/data/raw/images

Usage (test — final model only):
    python -m ml.evaluation.evaluate \\
        --checkpoint ml/experiments/resnet50/best.pt \\
        --split test \\
        --images_root ml/data/raw/images

Kaggle usage:
    python -m ml.evaluation.evaluate \\
        --checkpoint /kaggle/working/experiments/resnet50/best.pt \\
        --split val \\
        --images_root /kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1 \\
                      /kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2 \\
        --splits_csv /kaggle/working/data/splits.csv
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from ml.training.dataset import (
    SkinLesionDataset,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
)
from ml.training.augmentations import get_transforms

# ── Canonical class order ──────────────────────────────────────────────────────
CLASS_NAMES: List[str] = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
# akiec, bcc, bkl, df, mel, nv, vasc
ALL_LABELS: List[int] = list(range(NUM_CLASSES))


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, str, Dict]:
    """
    Loads a trained model from a best.pt checkpoint saved by train.py.

    The checkpoint stores:
        model_name        : timm architecture string (e.g. 'resnet50')
        model_state_dict  : trained weights
        config            : full training argparse dict

    Returns:
        (model, model_name, config_dict)
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_name = ckpt.get("model_name", "resnet50")
    config = ckpt.get("config", {})

    model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    best_epoch   = ckpt.get("epoch", "?")
    best_macro_f1 = ckpt.get("best_val_macro_f1", float("nan"))
    print(f"[INFO] Model: {model_name}  |  Best epoch: {best_epoch}  |  Saved val macro-F1: {best_macro_f1:.4f}")
    return model, model_name, config


# ==============================================================================
# Inference: collect logits, probabilities, predictions, and true labels
# ==============================================================================

def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the model over all batches.

    Returns:
        y_true     : (N,)   integer ground-truth labels
        y_pred     : (N,)   integer predicted labels (argmax)
        y_prob     : (N, C) softmax probabilities for AUC computation
    """
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
            else:
                logits = model(images)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())
            all_prob.append(probs.cpu().numpy())

    y_true = np.array(all_true, dtype=np.int64)
    y_pred = np.array(all_pred, dtype=np.int64)
    y_prob = np.vstack(all_prob)          # (N, 7)

    return y_true, y_pred, y_prob


# ==============================================================================
# Comprehensive Metrics
# ==============================================================================

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict:
    """
    Computes every requested metric and returns a structured dict.

    Sensitivity = Recall (TP / (TP + FN))  — per class
    Specificity = TN / (TN + FP)            — per class (one-vs-rest)
    AUC-ROC     = one-vs-rest per class     — requires probability scores
    """

    # ── Aggregate metrics ──────────────────────────────────────────────────
    acc        = float(accuracy_score(y_true, y_pred))
    bal_acc    = float(balanced_accuracy_score(y_true, y_pred))
    mac_prec   = float(precision_score(y_true, y_pred, average="macro",    zero_division=0))
    mac_rec    = float(recall_score   (y_true, y_pred, average="macro",    zero_division=0))
    mac_f1     = float(f1_score       (y_true, y_pred, average="macro",    zero_division=0))
    wt_prec    = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    wt_rec     = float(recall_score   (y_true, y_pred, average="weighted", zero_division=0))
    wt_f1      = float(f1_score       (y_true, y_pred, average="weighted", zero_division=0))
    mcc        = float(matthews_corrcoef(y_true, y_pred))

    # ── Macro AUC-ROC (one-vs-rest) ────────────────────────────────────────
    y_bin = label_binarize(y_true, classes=ALL_LABELS)   # (N, 7)
    try:
        mac_auc = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
    except ValueError:
        mac_auc = float("nan")

    # ── Per-class confusion counts and derived metrics ─────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)   # (7, 7)
    N  = cm.sum()

    per_class: Dict[str, Dict] = {}
    per_class_auc: Dict[str, float] = {}

    for i, cls in enumerate(CLASS_NAMES):
        TP = int(cm[i, i])
        FP = int(cm[:, i].sum() - TP)
        FN = int(cm[i, :].sum() - TP)
        TN = int(N - TP - FP - FN)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        ppv         = TP / (TP + FP) if (TP + FP) > 0 else 0.0   # precision
        npv         = TN / (TN + FN) if (TN + FN) > 0 else 0.0

        # Per-class AUC (one-vs-rest)
        try:
            auc_i = float(roc_auc_score(y_bin[:, i], y_prob[:, i]))
        except ValueError:
            auc_i = float("nan")
        per_class_auc[cls] = auc_i

        per_class[cls] = {
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "sensitivity": round(sensitivity, 4),   # same as recall
            "specificity": round(specificity, 4),
            "precision":   round(ppv,         4),
            "npv":         round(npv,          4),
            "auc_roc":     round(auc_i,        4),
        }

    # ── sklearn per-class P/R/F1 via classification_report ────────────────
    clf_dict = classification_report(
        y_true, y_pred,
        labels=ALL_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    clf_text = classification_report(
        y_true, y_pred,
        labels=ALL_LABELS,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    # Merge sklearn per-class P/R/F1 into per_class dict
    for cls in CLASS_NAMES:
        if cls in clf_dict:
            per_class[cls]["f1_score"] = round(clf_dict[cls]["f1-score"], 4)
            per_class[cls]["support"]  = int(clf_dict[cls]["support"])

    # ── Confusion matrix arrays ────────────────────────────────────────────
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1                    # avoid div-by-zero
    cm_norm = cm_norm / row_sums

    return {
        "aggregate": {
            "accuracy":           round(acc,      4),
            "balanced_accuracy":  round(bal_acc,   4),
            "macro_precision":    round(mac_prec,  4),
            "macro_recall":       round(mac_rec,   4),
            "macro_f1":           round(mac_f1,    4),
            "weighted_precision": round(wt_prec,   4),
            "weighted_recall":    round(wt_rec,    4),
            "weighted_f1":        round(wt_f1,     4),
            "macro_auc_roc":      round(mac_auc,   4),
            "mcc":                round(mcc,        4),
        },
        "per_class":          per_class,
        "per_class_auc":      per_class_auc,
        "confusion_matrix":   cm.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
        "classification_report_text": clf_text,
        "classification_report_dict": clf_dict,
    }


# ==============================================================================
# Melanoma analysis helper
# ==============================================================================

def write_melanoma_summary(metrics: Dict, output_path: Path, model_name: str, split: str) -> str:
    """
    Writes a markdown summary focused on melanoma recall and mel/nv confusion.
    Returns the markdown string.
    """
    cm    = np.array(metrics["confusion_matrix"])
    pc    = metrics["per_class"]
    agg   = metrics["aggregate"]

    mel_idx = CLASS_TO_IDX["mel"]   # 4
    nv_idx  = CLASS_TO_IDX["nv"]    # 5

    mel_total     = cm[mel_idx, :].sum()
    mel_correct   = cm[mel_idx, mel_idx]
    mel_as_nv     = cm[mel_idx, nv_idx]   # mel predicted as nv (dangerous miss)
    nv_as_mel     = cm[nv_idx, mel_idx]   # nv predicted as mel (false alarm)
    nv_total      = cm[nv_idx, :].sum()

    mel_sensitivity = pc["mel"]["sensitivity"]
    mel_specificity = pc["mel"]["specificity"]
    mel_auc         = pc["mel"]["auc_roc"]
    mel_precision   = pc["mel"]["precision"]
    nv_sensitivity  = pc["nv"]["sensitivity"]

    summary = f"""# Melanoma & Confusion Analysis — {model_name} ({split} split)

## Key Melanoma Metrics

| Metric | Value |
|---|---|
| Melanoma Sensitivity (Recall) | **{mel_sensitivity:.4f}** |
| Melanoma Specificity | {mel_specificity:.4f} |
| Melanoma Precision | {mel_precision:.4f} |
| Melanoma AUC-ROC (OvR) | {mel_auc:.4f} |
| Overall Macro-F1 | {agg["macro_f1"]:.4f} |
| Overall Balanced Accuracy | {agg["balanced_accuracy"]:.4f} |

> **Why melanoma recall matters:** A false negative (missed melanoma) is clinically
> far more dangerous than a false positive. We want sensitivity as close to 1.0 as
> possible, even if precision suffers.

## mel ↔ nv Confusion

| | Count | % of true class |
|---|---|---|
| mel correctly predicted as **mel** | {mel_correct} | {100*mel_correct/max(mel_total,1):.1f}% |
| mel **missed** and predicted as nv | **{mel_as_nv}** | **{100*mel_as_nv/max(mel_total,1):.1f}%** |
| mel predicted as other classes | {mel_total - mel_correct - mel_as_nv} | {100*(mel_total-mel_correct-mel_as_nv)/max(mel_total,1):.1f}% |
| nv falsely predicted as mel | {nv_as_mel} | {100*nv_as_mel/max(nv_total,1):.1f}% |

## Interpretation

"""
    # Generate automatic interpretation
    if mel_sensitivity >= 0.80:
        summary += "- ✅ **Melanoma recall is acceptable** (>= 0.80). The model catches most melanoma cases.\n"
    elif mel_sensitivity >= 0.60:
        summary += "- ⚠️ **Melanoma recall is moderate** (0.60–0.79). Consider class-weighted loss, oversampling, or focal loss tuning.\n"
    else:
        summary += "- ❌ **Melanoma recall is LOW** (< 0.60). The model misses most melanoma cases — this is the #1 priority to fix.\n"

    mel_nv_miss_pct = mel_as_nv / max(mel_total, 1)
    if mel_nv_miss_pct > 0.20:
        summary += (
            f"- ❌ **mel→nv confusion is high** ({100*mel_nv_miss_pct:.1f}% of melanoma predicted as benign nevi). "
            "Melanoma and common nevi share dermoscopic features (pigmentation patterns, borders). "
            "This is the most dangerous error type — a melanoma dismissed as a common mole.\n"
        )
    elif mel_nv_miss_pct > 0.10:
        summary += (
            f"- ⚠️ **mel→nv confusion is moderate** ({100*mel_nv_miss_pct:.1f}%). "
            "Some melanomas are predicted as benign nevi. "
            "Focal loss with higher gamma or a melanoma-specific sampling boost may reduce this.\n"
        )
    else:
        summary += f"- ✅ **mel→nv confusion is low** ({100*mel_nv_miss_pct:.1f}%).\n"

    nv_false_mel_pct = nv_as_mel / max(nv_total, 1)
    if nv_false_mel_pct > 0.10:
        summary += (
            f"- ⚠️ **nv→mel false alarms** ({100*nv_false_mel_pct:.1f}% of nevi predicted as melanoma). "
            "High false positive rate means unnecessary referrals but is clinically safer than missed mel.\n"
        )
    else:
        summary += f"- ✅ **nv→mel false alarm rate is low** ({100*nv_false_mel_pct:.1f}%).\n"

    summary += f"""
## All Per-Class Sensitivity / Specificity

| Class | Sensitivity | Specificity | AUC-ROC | Support |
|---|---|---|---|---|
"""
    for cls in CLASS_NAMES:
        p = pc[cls]
        flag = " ⚠️" if cls == "mel" and p["sensitivity"] < 0.75 else ""
        summary += (
            f"| **{cls}**{flag} | {p['sensitivity']:.4f} | {p['specificity']:.4f} "
            f"| {p['auc_roc']:.4f} | {p.get('support', '?')} |\n"
        )

    output_path.write_text(summary, encoding="utf-8")
    return summary


# ==============================================================================
# Plotting
# ==============================================================================

def plot_confusion_matrices(
    cm_counts: np.ndarray,
    cm_norm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str,
) -> None:
    """
    Saves side-by-side confusion matrices: raw counts and row-normalized (%).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, data, fmt, subtitle in [
        (axes[0], cm_counts, "d",    "Counts"),
        (axes[1], cm_norm,   ".2f",  "Normalized (row %)"),
    ]:
        im = ax.imshow(data, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=(1.0 if fmt == ".2f" else None))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            title=f"{title}\n({subtitle})",
            ylabel="True Diagnosis",
            xlabel="Predicted Diagnosis",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = data.max() / 2.0 if data.max() > 0 else 0.5
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j, i, format(data[i, j], fmt),
                    ha="center", va="center", fontsize=8,
                    color="white" if data[i, j] > thresh else "black",
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Confusion matrices -> {output_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str,
) -> None:
    """
    Plots per-class one-vs-rest ROC curves with AUC on one figure.
    Melanoma curve is drawn thicker and highlighted.
    """
    y_bin = label_binarize(y_true, classes=ALL_LABELS)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for i, cls in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc_val = roc_auc_score(y_bin[:, i], y_prob[:, i])
        except ValueError:
            continue

        lw    = 3.0 if cls == "mel" else 1.5
        alpha = 1.0 if cls == "mel" else 0.75
        label = f"{cls}  (AUC={auc_val:.3f})" + (" ← mel" if cls == "mel" else "")
        ax.plot(fpr, tpr, color=colors[i], lw=lw, alpha=alpha, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title(f"{title}\nPer-class One-vs-Rest ROC", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] ROC curves -> {output_path}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DermAI: Evaluate a trained model checkpoint.")

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to best.pt checkpoint (saved by train.py). E.g. ml/experiments/resnet50/best.pt",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "test"],
        help="Which split to evaluate on. Use 'test' ONLY for the final selected model (default: val).",
    )
    parser.add_argument(
        "--splits_csv", type=str, default="ml/data/splits.csv",
        help="Path to splits.csv (default: ml/data/splits.csv)",
    )
    parser.add_argument(
        "--images_root", type=str, nargs="+", default=["ml/data/raw/images"],
        help="Image folder(s). On Kaggle, pass both HAM10000_images_part_1 and part_2.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=(
            "Where to save evaluation results. "
            "Defaults to the same folder as the checkpoint."
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for inference (default: 32).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader workers (default: 0 for Windows).",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Compute device (default: auto).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = (device.type == "cuda")
    print(f"[INFO] Device: {device}  |  AMP: {use_amp}")

    # ── Test split guard ───────────────────────────────────────────────────
    if args.split == "test":
        print()
        print("=" * 70)
        print("  WARNING: You are evaluating on the TEST SPLIT.")
        print("  The test split must only be used ONCE, for the final")
        print("  chosen model. Do NOT use these results to pick models.")
        print("=" * 70)
        print()

    # ── Output directory ───────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if args.output_dir is None:
        output_dir = ckpt_path.parent
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.split}_eval"   # e.g. val_eval or test_eval

    # ── Load model ─────────────────────────────────────────────────────────
    model, model_name, _ = load_model(ckpt_path, device)

    # Count parameters
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameters: {total_params:,} total  |  {trainable_params:,} trainable")

    # ── DataLoader ─────────────────────────────────────────────────────────
    print(f"[INFO] Building {args.split} DataLoader from: {args.splits_csv}")
    dataset = SkinLesionDataset(
        splits_csv=args.splits_csv,
        images_roots=args.images_root,
        split=args.split,
        transform=get_transforms(args.split),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    class_counts = dataset.get_class_counts()
    print(f"[INFO] {args.split} split: {len(dataset)} images")
    print(f"       Per-class: { {k: v for k, v in class_counts.items()} }")

    # ── Inference ──────────────────────────────────────────────────────────
    print(f"\n[INFO] Running inference on {args.split} split...")
    y_true, y_pred, y_prob = run_inference(model, loader, device, use_amp=use_amp)

    # ── Metrics ────────────────────────────────────────────────────────────
    print("[INFO] Computing metrics...")
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    metrics["model"]           = model_name
    metrics["split"]           = args.split
    metrics["total_params"]    = total_params
    metrics["trainable_params"]= trainable_params
    metrics["checkpoint"]      = str(ckpt_path)
    metrics["num_images"]      = len(dataset)

    # ── Print summary ──────────────────────────────────────────────────────
    agg = metrics["aggregate"]
    print()
    print("=" * 65)
    print(f"  Evaluation Results: {model_name}  ({args.split} split)")
    print("=" * 65)
    print(f"  Accuracy          : {agg['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {agg['balanced_accuracy']:.4f}")
    print(f"  Macro Precision   : {agg['macro_precision']:.4f}")
    print(f"  Macro Recall      : {agg['macro_recall']:.4f}")
    print(f"  Macro F1          : {agg['macro_f1']:.4f}")
    print(f"  Weighted F1       : {agg['weighted_f1']:.4f}")
    print(f"  Macro AUC-ROC     : {agg['macro_auc_roc']:.4f}")
    print(f"  MCC               : {agg['mcc']:.4f}")
    print()
    print(f"  {'Class':<8} {'Sensitiv':<10} {'Specific':<10} {'Precision':<11} {'AUC-ROC':<10} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6}")
    print("  " + "-" * 75)
    for cls in CLASS_NAMES:
        p = metrics["per_class"][cls]
        flag = " <--" if cls == "mel" else ""
        print(
            f"  {cls:<8} {p['sensitivity']:<10.4f} {p['specificity']:<10.4f} "
            f"{p['precision']:<11.4f} {p['auc_roc']:<10.4f} "
            f"{p['TP']:<6} {p['FP']:<6} {p['TN']:<6} {p['FN']:<6}{flag}"
        )
    print()
    print("  Classification Report:")
    print(metrics["classification_report_text"])

    # ── Save artifacts ─────────────────────────────────────────────────────
    # 1) Metrics JSON
    metrics_save = {k: v for k, v in metrics.items()
                    if k not in ("confusion_matrix", "confusion_matrix_norm",
                                  "classification_report_dict")}
    metrics_json_path = output_dir / f"{prefix}_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    print(f"[SAVED] Metrics JSON   -> {metrics_json_path}")

    # 2) Classification report text
    report_path = output_dir / f"{prefix}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}  |  Split: {args.split}\n")
        f.write(f"Checkpoint: {ckpt_path}\n\n")
        f.write(metrics["classification_report_text"])
        f.write("\nAggregate Metrics:\n")
        for k, v in agg.items():
            f.write(f"  {k}: {v}\n")
    print(f"[SAVED] Report text    -> {report_path}")

    # 3) Confusion matrices (counts + normalized)
    cm_counts = np.array(metrics["confusion_matrix"])
    cm_norm   = np.array(metrics["confusion_matrix_norm"])
    plot_confusion_matrices(
        cm_counts, cm_norm, CLASS_NAMES,
        output_path=output_dir / f"{prefix}_confusion_matrix.png",
        title=f"{model_name} — {args.split} split",
    )

    # 4) ROC curves
    plot_roc_curves(
        y_true, y_prob, CLASS_NAMES,
        output_path=output_dir / f"{prefix}_roc_curves.png",
        title=f"{model_name} — {args.split} split",
    )

    # 5) Melanoma summary markdown
    mel_md_path = output_dir / f"{prefix}_melanoma_summary.md"
    write_melanoma_summary(metrics, mel_md_path, model_name, args.split)
    print(f"[SAVED] Melanoma MD    -> {mel_md_path}")

    print(f"\n[DONE] All evaluation artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()

