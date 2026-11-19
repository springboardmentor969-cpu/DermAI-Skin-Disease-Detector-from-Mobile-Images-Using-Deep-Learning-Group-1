"""
Confidence Calibration and Selective Classification Threshold Analysis.

This module evaluates model confidence calibration on the VALIDATION split ONLY:
1. Computes raw softmax probabilities on the validation split.
2. Computes Expected Calibration Error (ECE) and plots a reliability diagram.
3. Fits temperature scaling (Guo et al., 2017) on validation logits and reports ECE before and after.
4. Sweeps confidence thresholds from 0.40 to 0.90 in steps of 0.05, reporting:
   - Coverage (percentage of cases where max probability >= threshold)
   - Accuracy on accepted cases
   - Macro-F1 on accepted cases
   - Melanoma (mel) recall on accepted cases
5. Generates and saves:
   - Reliability diagrams (before & after temperature scaling)
   - Confidence threshold trade-off curves (coverage vs accuracy / macro-F1 / mel recall)
   - Markdown and CSV summary tables

IMPORTANT CLINICAL SAFETY NOTICE:
- The final CONFIDENCE_THRESHOLD is a project-defined operating point balancing
  automated screening coverage against the safety of mandatory clinical escalation.
- High thresholds improve accepted case reliability but increase physician review workload.
- Low thresholds improve coverage but risk accepting uncertain classifications.
- Empirical metrics on the validation split are reported transparently without claiming target accuracy.
- STRICT RULE: The test split is NOT used in this module to prevent data leakage.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.dataset import SkinLesionDataset, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES
from ml.training.augmentations import get_transforms
from ml.evaluation.evaluate import load_model, CLASS_NAMES


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) and returns per-bin statistics.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    n_samples = len(confidences)
    ece = 0.0
    bin_details = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples within this confidence bin
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(predictions[in_bin] == targets[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / n_samples) * bin_error
        else:
            bin_acc = 0.0
            bin_conf = (bin_lower + bin_upper) / 2.0
            bin_error = 0.0

        bin_details.append({
            "bin_idx": i,
            "bin_lower": float(bin_lower),
            "bin_upper": float(bin_upper),
            "count": bin_count,
            "fraction": float(bin_count / n_samples),
            "accuracy": bin_acc,
            "confidence": bin_conf,
            "error": bin_error,
        })

    return float(ece), bin_details


def fit_temperature_scaling(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lr: float = 0.02,
    max_iter: int = 100,
) -> float:
    """
    Learns an optimal scalar temperature T > 0 using L-BFGS to minimize
    Negative Log-Likelihood (cross-entropy) on validation logits.
    """
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def eval_loss():
        optimizer.zero_grad()
        # Constrain temperature to remain strictly positive
        temp_clamped = torch.clamp(temperature, min=0.01)
        scaled_logits = logits / temp_clamped
        loss = criterion(scaled_logits, targets)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    optimal_t = float(torch.clamp(temperature, min=0.01).item())
    return optimal_t


@torch.no_grad()
def collect_validation_logits_and_targets(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs inference across the validation loader and aggregates raw logits and targets."""
    model.eval()
    all_logits = []
    all_targets = []

    for images, labels, _ in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def evaluate_threshold_sweep(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    mel_class_idx: int,
    thresholds: List[float],
) -> pd.DataFrame:
    """
    Evaluates selective classification performance at each confidence threshold.
    """
    n_total = len(targets)
    total_true_mel = int(np.sum(targets == mel_class_idx))
    rows = []

    for tau in thresholds:
        accepted_mask = confidences >= tau
        n_accepted = int(np.sum(accepted_mask))
        coverage = n_accepted / n_total

        if n_accepted == 0:
            rows.append({
                "threshold": tau,
                "coverage": 0.0,
                "accepted_count": 0,
                "deferred_count": n_total,
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "mel_recall_accepted": np.nan,
                "mel_cases_accepted": 0,
                "mel_detected_accepted": 0,
            })
            continue

        acc_targets = targets[accepted_mask]
        acc_preds = predictions[accepted_mask]

        acc = float(accuracy_score(acc_targets, acc_preds))
        f1_macro = float(f1_score(acc_targets, acc_preds, average="macro", zero_division=0))

        # Melanoma recall on accepted cases
        mel_in_accepted = int(np.sum(acc_targets == mel_class_idx))
        mel_detected = int(np.sum((acc_targets == mel_class_idx) & (acc_preds == mel_class_idx)))

        mel_recall_acc = (
            float(mel_detected / mel_in_accepted)
            if mel_in_accepted > 0
            else np.nan
        )

        rows.append({
            "threshold": tau,
            "coverage": round(coverage, 4),
            "accepted_count": n_accepted,
            "deferred_count": n_total - n_accepted,
            "accuracy": round(acc, 4),
            "macro_f1": round(f1_macro, 4),
            "mel_recall_accepted": round(mel_recall_acc, 4) if not np.isnan(mel_recall_acc) else np.nan,
            "mel_cases_accepted": mel_in_accepted,
            "mel_detected_accepted": mel_detected,
        })

    return pd.DataFrame(rows)


def plot_reliability_diagrams(
    bins_uncal: List[Dict[str, Any]],
    ece_uncal: float,
    bins_cal: List[Dict[str, Any]],
    ece_cal: float,
    optimal_temp: float,
    save_path: Path,
) -> None:
    """Plots side-by-side reliability diagrams before and after temperature scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, bin_data, ece_val, title in zip(
        axes,
        [bins_uncal, bins_cal],
        [ece_uncal, ece_cal],
        [
            f"Uncalibrated (ECE = {ece_uncal:.4f})",
            f"Temperature-Scaled (T = {optimal_temp:.2f}, ECE = {ece_cal:.4f})",
        ],
    ):
        confs = [b["confidence"] for b in bin_data]
        accs = [b["accuracy"] for b in bin_data]
        counts = [b["count"] for b in bin_data]
        centers = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bin_data]
        width = 1.0 / len(bin_data)

        # Plot bars
        ax.bar(centers, accs, width=width * 0.9, alpha=0.7, color="#0284c7", edgecolor="#0369a1", label="Outputs")

        # Plot calibration gaps
        for c, a, conf in zip(centers, accs, confs):
            if abs(a - conf) > 1e-4:
                ax.plot([c, c], [a, conf], color="#e11d48", lw=2, alpha=0.7)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "--", color="#64748b", lw=1.5, label="Perfect Calibration")

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Mean Predicted Confidence", fontsize=10)
        ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_threshold_tradeoff(
    df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Plots trade-off curves for Coverage, Accepted Accuracy, Macro-F1, and Melanoma Recall."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    taus = df["threshold"].values

    ax.plot(taus, df["coverage"] * 100, "o-", color="#0284c7", lw=2, label="Coverage (% cases accepted)")
    ax.plot(taus, df["accuracy"] * 100, "s-", color="#16a34a", lw=2, label="Accuracy on Accepted Cases (%)")
    ax.plot(taus, df["macro_f1"] * 100, "^-", color="#9333ea", lw=2, label="Macro-F1 on Accepted Cases (%)")
    ax.plot(taus, df["mel_recall_accepted"] * 100, "d-", color="#e11d48", lw=2, label="Melanoma Recall on Accepted (%)")

    ax.set_title("Selective Classification Trade-off across Confidence Thresholds (Validation Split)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Confidence Threshold (tau)", fontsize=10)
    ax.set_ylabel("Metric Value (%)", fontsize=10)
    ax.set_xticks(taus)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def run_calibration(
    checkpoint_path: Path,
    splits_csv: Path,
    images_root: Path,
    output_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    device_str: Optional[str] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("\n" + "=" * 75)
    print("  CONFIDENCE CALIBRATION & THRESHOLD ANALYSIS (VALIDATION SPLIT ONLY)")
    print("=" * 75)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Splits CSV: {splits_csv}")
    print("[INFO] Evaluating strictly on VALIDATION split. Test split is untouched.\n")

    # 1. Load model dynamically
    model, model_name, model_config = load_model(Path(checkpoint_path), device=device)
    model.eval()

    # 2. Build Validation DataLoader strictly
    val_transform = get_transforms("val", img_size=224)
    val_dataset = SkinLesionDataset(
        splits_csv=splits_csv,
        images_roots=images_root,
        split="val",  # STRICT VALIDATION SPLIT ONLY
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    print(f"[INFO] Loaded {len(val_dataset)} validation samples.")

    # 3. Compute raw validation logits and targets
    logits, targets = collect_validation_logits_and_targets(model, val_loader, device=device)
    targets_np = targets.numpy()

    # Uncalibrated probabilities & confidences
    probs_uncal = F.softmax(logits, dim=1).numpy()
    preds_uncal = np.argmax(probs_uncal, axis=1)
    confs_uncal = np.max(probs_uncal, axis=1)

    # 4. Compute uncalibrated ECE and reliability bins
    ece_uncal, bins_uncal = compute_ece(confs_uncal, preds_uncal, targets_np, num_bins=10)

    # 5. Fit temperature scaling on the validation split
    optimal_temp = fit_temperature_scaling(logits, targets)

    # Calibrated probabilities & confidences
    scaled_logits = logits / optimal_temp
    probs_cal = F.softmax(scaled_logits, dim=1).numpy()
    preds_cal = np.argmax(probs_cal, axis=1)
    confs_cal = np.max(probs_cal, axis=1)

    ece_cal, bins_cal = compute_ece(confs_cal, preds_cal, targets_np, num_bins=10)

    print("-" * 75)
    print(f"  Expected Calibration Error (ECE) before temperature scaling: {ece_uncal:.4f}")
    print(f"  Fitted Temperature Parameter (T*):                          {optimal_temp:.4f}")
    print(f"  Expected Calibration Error (ECE) after temperature scaling:  {ece_cal:.4f}")
    print(f"  ECE Reduction:                                              {(ece_uncal - ece_cal):.4f}")
    print("-" * 75)

    # 6. Save reliability diagrams
    diag_path = output_dir / "calibration_reliability_diagrams.png"
    plot_reliability_diagrams(bins_uncal, ece_uncal, bins_cal, ece_cal, optimal_temp, diag_path)
    print(f"[SAVED] Reliability diagrams -> {diag_path}")

    # 7. Sweep confidence thresholds from 0.40 to 0.90 in steps of 0.05
    thresholds = [round(t, 2) for t in np.arange(0.40, 0.95, 0.05)]
    mel_class_idx = CLASS_TO_IDX.get("mel", 4)

    # Evaluate sweep on both uncalibrated and calibrated probabilities
    df_cal = evaluate_threshold_sweep(confs_cal, preds_cal, targets_np, mel_class_idx, thresholds)
    df_uncal = evaluate_threshold_sweep(confs_uncal, preds_uncal, targets_np, mel_class_idx, thresholds)

    # Print calibrated threshold sweep table
    print("\n==========================================================================================")
    print("  CONFIDENCE THRESHOLD SWEEP TABLE (Calibrated Probabilities - Validation Split)")
    print("==========================================================================================")
    print(f"{'Threshold':>9} | {'Coverage':>8} | {'Accepted':>8} | {'Deferred':>8} | {'Acc (Accepted)':>14} | {'Macro-F1':>8} | {'Mel Recall':>10}")
    print("-" * 90)
    for _, row in df_cal.iterrows():
        acc_str = f"{row['accuracy']:.4f}" if not pd.isna(row['accuracy']) else "N/A"
        f1_str = f"{row['macro_f1']:.4f}" if not pd.isna(row['macro_f1']) else "N/A"
        mel_str = f"{row['mel_recall_accepted']:.4f}" if not pd.isna(row['mel_recall_accepted']) else "N/A"
        print(f"{row['threshold']:9.2f} | {row['coverage']:8.4f} | {int(row['accepted_count']):8d} | {int(row['deferred_count']):8d} | {acc_str:>14} | {f1_str:>8} | {mel_str:>10}")
    print("=" * 90)

    # 8. Save CSV and Markdown
    csv_path = output_dir / "confidence_threshold_sweep.csv"
    df_cal.to_csv(csv_path, index=False)
    print(f"[SAVED] Threshold sweep CSV -> {csv_path}")

    tradeoff_chart_path = output_dir / "confidence_threshold_tradeoff.png"
    plot_threshold_tradeoff(df_cal, tradeoff_chart_path)
    print(f"[SAVED] Threshold trade-off curves -> {tradeoff_chart_path}")

    # Build Markdown report
    md_content = f"""# DermAI Confidence Calibration & Threshold Analysis

> **Protocol Note**: Evaluated strictly on the **VALIDATION split** ({len(val_dataset)} samples).
> The **test split is NOT used** in this analysis to prevent data leakage.

## Temperature Scaling Calibration
- **Uncalibrated ECE**: `{ece_uncal:.4f}`
- **Learned Temperature Parameter ($T^*$):** `{optimal_temp:.4f}`
- **Calibrated ECE**: `{ece_cal:.4f}`
- **ECE Improvement**: `{ece_uncal - ece_cal:.4f}`

## Selective Classification Threshold Sweep (Calibrated)

| Threshold ($\\tau$) | Coverage | Accepted Cases | Deferred to Clinician | Accuracy (Accepted) | Macro-F1 (Accepted) | Mel Recall (Accepted) |
|---|---|---|---|---|---|---|
"""
    for _, row in df_cal.iterrows():
        acc_str = f"{row['accuracy']:.4f}" if not pd.isna(row['accuracy']) else "N/A"
        f1_str = f"{row['macro_f1']:.4f}" if not pd.isna(row['macro_f1']) else "N/A"
        mel_str = f"{row['mel_recall_accepted']:.4f}" if not pd.isna(row['mel_recall_accepted']) else "N/A"
        md_content += f"| `{row['threshold']:.2f}` | `{row['coverage'] * 100:.1f}%` | `{int(row['accepted_count'])}` | `{int(row['deferred_count'])}` | `{acc_str}` | `{f1_str}` | `{mel_str}` |\n"

    md_content += r"""
---

## Project-Defined Choice & Clinical Trade-off Documentation

> [!IMPORTANT]
> The final `CONFIDENCE_THRESHOLD` is a **project-defined operational choice** balancing
> decision-support utility against clinician review workload.
>
> 1. **High Thresholds ($\tau \ge 0.70$ - $0.80$):**
>    - Higher reliability and accuracy on accepted predictions.
>    - Trade-off: Lower coverage (more cases deferred), meaning higher review burden on clinicians.
>
> 2. **Moderate Thresholds ($\tau \approx 0.60$):**
>    - Balances reasonable screening coverage with safety.
>    - Filters out uncertain border presentations while providing AI assistance on clear cases.
>
> 3. **Low Thresholds ($\tau \le 0.45$):**
>    - High coverage, but accepts borderline and potentially misleading predictions without clinician review.
>
> *No target accuracy is claimed; the threshold should be selected based on clinical triage protocol and available specialist review capacity.*
"""
    md_path = output_dir / "confidence_threshold_sweep.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"[SAVED] Calibration Markdown -> {md_path}")

    # Save JSON summary
    summary = {
        "dataset": "HAM10000_validation_split",
        "num_val_samples": len(val_dataset),
        "uncalibrated_ece": float(ece_uncal),
        "optimal_temperature": float(optimal_temp),
        "calibrated_ece": float(ece_cal),
        "ece_reduction": float(ece_uncal - ece_cal),
        "calibrated_thresholds": df_cal.to_dict(orient="records"),
        "uncalibrated_thresholds": df_uncal.to_dict(orient="records"),
    }
    json_path = output_dir / "calibration_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] Calibration JSON -> {json_path}")
    print("\n[DONE] Confidence calibration analysis complete.")


def main():
    parser = argparse.ArgumentParser(description="Calibrate confidence and analyze selective classification thresholds.")
    parser.add_argument("--checkpoint", type=str, default="ml/experiments/resnet50_smoke/best.pt", help="Path to checkpoint best.pt.")
    parser.add_argument("--splits_csv", type=str, default="ml/data/splits.csv", help="Path to splits.csv.")
    parser.add_argument("--images_root", type=str, default="ml/data/raw/images", help="Path to images directory.")
    parser.add_argument("--output_dir", type=str, default="ml/experiments/confidence_calibration", help="Output directory for calibration artifacts.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu).")
    args = parser.parse_args()

    run_calibration(
        checkpoint_path=Path(args.checkpoint),
        splits_csv=Path(args.splits_csv),
        images_root=Path(args.images_root),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
