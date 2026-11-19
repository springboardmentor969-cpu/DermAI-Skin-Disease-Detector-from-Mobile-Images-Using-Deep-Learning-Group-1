"""
DermAI Milestone 2: Multi-Model Comparison Script.

Evaluates every trained model found in ml/experiments/ on the VALIDATION split,
builds a comparison table, and identifies the best model by val macro-F1.

Outputs (all in ml/experiments/):
  model_comparison.csv          - Full metric table (CSV)
  model_comparison.md           - Markdown table (best model highlighted)
  model_comparison_chart.png    - Bar chart: macro-F1 and balanced accuracy
  final_test_results.json       - ONLY produced when you pass --run_test for
                                  the single best model. Heavy warning shown.

RULES:
  1. Validation only — never use the test split to pick models.
  2. The test split is run at most once, for the final chosen model, and only
     when --run_test is explicitly passed.
  3. Training time is read from history.csv (total seconds across all epochs),
     not from wall-clock re-measurement.

Usage (compare all models on val):
    python -m ml.evaluation.compare_models \\
        --experiments_dir ml/experiments \\
        --images_root ml/data/raw/images

    On Kaggle (after downloading experiments/):
    python -m ml.evaluation.compare_models \\
        --experiments_dir /kaggle/working/experiments \\
        --images_root /kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1 \\
                      /kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2 \\
        --splits_csv /kaggle/working/data/splits.csv

Run test on best model ONLY after all validation comparisons are done:
    python -m ml.evaluation.compare_models \\
        --experiments_dir ml/experiments \\
        --images_root ml/data/raw/images \\
        --run_test
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from ml.training.dataset import (
    SkinLesionDataset,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
)
from ml.training.augmentations import get_transforms
from ml.evaluation.evaluate import (
    load_model,
    run_inference,
    compute_all_metrics,
    plot_confusion_matrices,
    plot_roc_curves,
    write_melanoma_summary,
    CLASS_NAMES,
    ALL_LABELS,
)


# ==============================================================================
# Discover model experiment folders
# ==============================================================================

def find_model_dirs(experiments_dir: Path) -> List[Path]:
    """
    Returns every sub-directory of experiments_dir that contains a best.pt file.
    Skips directories named 'augmentation_grid*' (check_data output).
    """
    dirs = []
    if not experiments_dir.is_dir():
        return dirs
    for subdir in sorted(experiments_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name.startswith("augmentation"):
            continue
        if (subdir / "best.pt").is_file():
            dirs.append(subdir)
    return dirs


def read_training_time(model_dir: Path) -> Optional[float]:
    """
    Reads training time (seconds) from history.csv, or from config.json.
    Returns None if not available.
    """
    hist_path = model_dir / "history.csv"
    if hist_path.is_file():
        try:
            hist = pd.read_csv(hist_path)
            if "time_sec" in hist.columns:
                return float(hist["time_sec"].sum())
        except Exception:
            pass

    # Fallback: try reading from metrics.json training_time field
    metrics_path = model_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                m = json.load(f)
            return m.get("training_time_seconds")
        except Exception:
            pass
    return None


def count_parameters(model) -> Tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ==============================================================================
# Evaluate a single model on a split — returns metrics dict
# ==============================================================================

def evaluate_model_on_split(
    model_dir: Path,
    split: str,
    splits_csv: str,
    images_root: List[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
    save_plots: bool = True,
) -> Optional[Dict]:
    """
    Loads best.pt from model_dir, runs inference, returns full metrics dict.
    Also saves confusion matrix + ROC curves inside model_dir if save_plots=True.
    """
    ckpt_path = model_dir / "best.pt"
    try:
        model, model_name, _ = load_model(ckpt_path, device)
    except Exception as e:
        print(f"  [SKIP] Could not load {ckpt_path}: {e}")
        return None

    total_params, _ = count_parameters(model)

    dataset = SkinLesionDataset(
        splits_csv=splits_csv,
        images_roots=images_root,
        split=split,
        transform=get_transforms(split),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    y_true, y_pred, y_prob = run_inference(model, loader, device, use_amp=use_amp)
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    metrics["model"]        = model_name
    metrics["split"]        = split
    metrics["total_params"] = total_params
    metrics["num_images"]   = len(dataset)

    if save_plots:
        prefix = f"{split}_eval"
        cm_counts = np.array(metrics["confusion_matrix"])
        cm_norm   = np.array(metrics["confusion_matrix_norm"])
        plot_confusion_matrices(
            cm_counts, cm_norm, CLASS_NAMES,
            output_path=model_dir / f"{prefix}_confusion_matrix.png",
            title=f"{model_name} — {split} split",
        )
        plot_roc_curves(
            y_true, y_prob, CLASS_NAMES,
            output_path=model_dir / f"{prefix}_roc_curves.png",
            title=f"{model_name} — {split} split",
        )
        mel_md_path = model_dir / f"{prefix}_melanoma_summary.md"
        write_melanoma_summary(metrics, mel_md_path, model_name, split)

        # Save per-model eval metrics
        with open(model_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2,
                      default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    return metrics


# ==============================================================================
# Build comparison table
# ==============================================================================

def build_comparison_table(
    results: List[Dict],        # list of metrics dicts, one per model
    training_times: Dict[str, Optional[float]],
) -> pd.DataFrame:
    """
    Assembles a flat DataFrame with one row per model and all key metrics.
    """
    rows = []
    for m in results:
        agg  = m["aggregate"]
        pc   = m["per_class"]
        name = m["model"]

        row = {
            "model":               name,
            "params_M":            round(m.get("total_params", 0) / 1e6, 2),
            "train_time_min":      round(training_times.get(name, 0) / 60, 1)
                                   if training_times.get(name) else None,
            "accuracy":            agg["accuracy"],
            "balanced_accuracy":   agg["balanced_accuracy"],
            "macro_f1":            agg["macro_f1"],
            "weighted_f1":         agg["weighted_f1"],
            "macro_precision":     agg["macro_precision"],
            "macro_recall":        agg["macro_recall"],
            "weighted_precision":  agg["weighted_precision"],
            "weighted_recall":     agg["weighted_recall"],
            "macro_auc_roc":       agg["macro_auc_roc"],
            "mcc":                 agg["mcc"],
            # Per-class mel metrics (clinically critical)
            "mel_sensitivity":     pc["mel"]["sensitivity"],
            "mel_specificity":     pc["mel"]["specificity"],
            "mel_precision":       pc["mel"]["precision"],
            "mel_auc_roc":         pc["mel"]["auc_roc"],
            "mel_f1":              pc["mel"].get("f1_score", float("nan")),
        }
        # Per-class F1 for all classes
        for cls in CLASS_NAMES:
            row[f"{cls}_f1"] = pc[cls].get("f1_score", float("nan"))

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return df


# ==============================================================================
# Markdown table with best model highlighted
# ==============================================================================

def build_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No models evaluated._\n"

    best_model = df.iloc[0]["model"]

    lines = [
        "# DermAI Milestone 2 — Validation Model Comparison",
        "",
        "> Models ranked by **val macro-F1** (higher is better).",
        "> The **best model** (highlighted with ✅) should then be evaluated",
        "> once on the test split using `--run_test`.",
        "",
        "## Summary Table",
        "",
        "| Model | Params (M) | Train (min) | Accuracy | Bal-Acc | Macro-F1 | Wt-F1 | Macro-AUC | MCC | mel-Sensitivity | mel-AUC |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for _, row in df.iterrows():
        tag = "✅ **" if row["model"] == best_model else ""
        end = "**" if row["model"] == best_model else ""
        train_min = f"{row['train_time_min']:.1f}" if row["train_time_min"] else "—"
        lines.append(
            f"| {tag}{row['model']}{end} "
            f"| {row['params_M']} "
            f"| {train_min} "
            f"| {row['accuracy']:.4f} "
            f"| {row['balanced_accuracy']:.4f} "
            f"| {row['macro_f1']:.4f} "
            f"| {row['weighted_f1']:.4f} "
            f"| {row['macro_auc_roc']:.4f} "
            f"| {row['mcc']:.4f} "
            f"| {row['mel_sensitivity']:.4f} "
            f"| {row['mel_auc_roc']:.4f} |"
        )

    lines += [
        "",
        "## Per-Class F1 (val)",
        "",
        "| Model | akiec | bcc | bkl | df | mel | nv | vasc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        tag = "✅ **" if row["model"] == best_model else ""
        end = "**" if row["model"] == best_model else ""
        cls_cells = " | ".join(
            f"{row.get(f'{cls}_f1', float('nan')):.4f}" for cls in CLASS_NAMES
        )
        lines.append(f"| {tag}{row['model']}{end} | {cls_cells} |")

    lines += [
        "",
        "---",
        f"**Best model by val macro-F1:** `{best_model}`",
        "",
        "> [!IMPORTANT]",
        "> Run the test split **only once** for `{best_model}` using:",
        "> ```bash",
        "> python -m ml.evaluation.compare_models --run_test ...",
        "> ```",
        "",
    ]

    return "\n".join(lines) + "\n"


# ==============================================================================
# Bar chart: macro-F1 and balanced accuracy per model
# ==============================================================================

def plot_comparison_chart(df: pd.DataFrame, output_path: Path) -> None:
    """
    Grouped bar chart comparing macro-F1, balanced accuracy, macro AUC-ROC,
    and melanoma sensitivity across all evaluated models.
    """
    if df.empty:
        return

    models   = df["model"].tolist()
    x        = np.arange(len(models))
    width    = 0.2

    fig, ax = plt.subplots(figsize=(max(8, 3 * len(models)), 6))

    bars_f1  = ax.bar(x - 1.5 * width, df["macro_f1"],          width, label="Macro-F1",          color="#4c72b0")
    bars_ba  = ax.bar(x - 0.5 * width, df["balanced_accuracy"],  width, label="Balanced Accuracy",  color="#dd8452")
    bars_auc = ax.bar(x + 0.5 * width, df["macro_auc_roc"],      width, label="Macro AUC-ROC",      color="#55a868")
    bars_mel = ax.bar(x + 1.5 * width, df["mel_sensitivity"],    width, label="mel Sensitivity",    color="#c44e52")

    # Annotate best model
    best_idx = 0
    ax.axvline(x=best_idx, color="gold", linestyle="--", alpha=0.6, linewidth=2, label="Best model")

    for bars in [bars_f1, bars_ba, bars_auc, bars_mel]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=90
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "DermAI — Validation Metrics by Architecture\n"
        "(Sorted by Macro-F1, best model marked with dashed line)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Comparison chart -> {output_path}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DermAI: Compare trained models on the validation split, then optionally run the test split."
    )
    parser.add_argument(
        "--experiments_dir", type=str, default="ml/experiments",
        help="Root experiments directory containing one sub-folder per model (default: ml/experiments).",
    )
    parser.add_argument(
        "--splits_csv", type=str, default="ml/data/splits.csv",
        help="Path to splits.csv (default: ml/data/splits.csv).",
    )
    parser.add_argument(
        "--images_root", type=str, nargs="+", default=["ml/data/raw/images"],
        help="Image folder(s). On Kaggle pass both HAM10000_images_part_1 and part_2.",
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
    parser.add_argument(
        "--run_test", action="store_true", default=False,
        help=(
            "After validation comparison, run the test split ONCE on the best model. "
            "ONLY use this flag after you have decided on the final model. "
            "Results are saved to ml/experiments/final_test_results.json."
        ),
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

    experiments_dir = Path(args.experiments_dir)
    model_dirs = find_model_dirs(experiments_dir)

    if not model_dirs:
        print(f"[ERROR] No model sub-directories with best.pt found in: {experiments_dir}")
        print("        Train at least one model first with ml/training/train.py")
        sys.exit(1)

    print(f"\n[INFO] Found {len(model_dirs)} trained model(s):")
    for d in model_dirs:
        print(f"  - {d.name}")

    # ── Validate on all models ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VALIDATION SPLIT EVALUATION (used for model selection)")
    print("=" * 70)

    val_results  : List[Dict] = []
    training_times: Dict[str, Optional[float]] = {}

    for mdir in model_dirs:
        print(f"\n--- Evaluating: {mdir.name} ---")
        t = read_training_time(mdir)
        metrics = evaluate_model_on_split(
            model_dir=mdir,
            split="val",
            splits_csv=args.splits_csv,
            images_root=args.images_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            use_amp=use_amp,
            save_plots=True,
        )
        if metrics is None:
            continue

        val_results.append(metrics)
        training_times[metrics["model"]] = t

        agg = metrics["aggregate"]
        print(
            f"  macro-F1={agg['macro_f1']:.4f}  "
            f"bal-acc={agg['balanced_accuracy']:.4f}  "
            f"AUC={agg['macro_auc_roc']:.4f}  "
            f"mel-sensitivity={metrics['per_class']['mel']['sensitivity']:.4f}"
        )

    if not val_results:
        print("[ERROR] No models evaluated successfully.")
        sys.exit(1)

    # ── Build comparison table ─────────────────────────────────────────────
    df = build_comparison_table(val_results, training_times)
    best_row  = df.iloc[0]
    best_model_name = best_row["model"]

    print("\n" + "=" * 70)
    print(f"  BEST MODEL (by val macro-F1): {best_model_name}")
    print(f"  Macro-F1 = {best_row['macro_f1']:.4f}  |  "
          f"Bal-Acc = {best_row['balanced_accuracy']:.4f}  |  "
          f"mel-Sensitivity = {best_row['mel_sensitivity']:.4f}")
    print("=" * 70)

    # ── Save comparison artifacts ──────────────────────────────────────────
    csv_path = experiments_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Comparison CSV   -> {csv_path}")

    md_str = build_markdown_table(df)
    md_path = experiments_dir / "model_comparison.md"
    md_path.write_text(md_str, encoding="utf-8")
    print(f"[SAVED] Comparison MD    -> {md_path}")

    chart_path = experiments_dir / "model_comparison_chart.png"
    plot_comparison_chart(df, chart_path)

    # Print markdown table to console for easy reading
    print("\n" + md_str.encode('ascii', errors='ignore').decode('ascii'))

    # ── Optional: test split on best model ────────────────────────────────
    if args.run_test:
        print()
        print("=" * 70)
        print("  *** TEST SPLIT EVALUATION ***")
        print()
        print("  WARNING: The test split must only be used ONCE, for the FINAL")
        print("  chosen model. Do NOT use these results to compare or re-select")
        print("  models — that is data leakage. You have been warned.")
        print("=" * 70)
        print()

        # Find the dir for the best model
        best_dir = None
        for mdir in model_dirs:
            # The model_name inside best.pt may differ from the folder name
            # (e.g. folder 'resnet50_smoke' vs model 'resnet50').
            # Match on the model name we stored.
            for m in val_results:
                if m["model"] == best_model_name:
                    # Find whose folder produced this result
                    pass
            if mdir.name == best_model_name or any(
                r["model"] == best_model_name and str(mdir) in str(r.get("checkpoint", ""))
                for r in val_results
            ):
                best_dir = mdir
                break

        # Simpler fallback: just use the best.pt that gave best val score
        if best_dir is None:
            for mdir in model_dirs:
                ckpt = mdir / "best.pt"
                if ckpt.is_file():
                    try:
                        ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
                        if ckpt_data.get("model_name") == best_model_name:
                            best_dir = mdir
                            break
                    except Exception:
                        continue

        if best_dir is None:
            print(f"[ERROR] Could not locate experiment folder for best model '{best_model_name}'.")
            sys.exit(1)

        print(f"[INFO] Running test split evaluation on: {best_dir}")
        test_metrics = evaluate_model_on_split(
            model_dir=best_dir,
            split="test",
            splits_csv=args.splits_csv,
            images_root=args.images_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            use_amp=use_amp,
            save_plots=True,
        )

        if test_metrics is None:
            print("[ERROR] Test evaluation failed.")
            sys.exit(1)

        t_agg = test_metrics["aggregate"]
        print()
        print("  Test Results:")
        print(f"  Accuracy         : {t_agg['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {t_agg['balanced_accuracy']:.4f}")
        print(f"  Macro-F1         : {t_agg['macro_f1']:.4f}")
        print(f"  Macro AUC-ROC    : {t_agg['macro_auc_roc']:.4f}")
        print(f"  MCC              : {t_agg['mcc']:.4f}")
        print(f"  mel Sensitivity  : {test_metrics['per_class']['mel']['sensitivity']:.4f}")
        print()
        print(test_metrics["classification_report_text"])

        # Save final test results
        final_payload = {
            "best_model": best_model_name,
            "best_model_dir": str(best_dir),
            "val_macro_f1": float(best_row["macro_f1"]),
            "val_balanced_accuracy": float(best_row["balanced_accuracy"]),
            "test_results": {
                "aggregate": test_metrics["aggregate"],
                "per_class": test_metrics["per_class"],
                "classification_report_text": test_metrics["classification_report_text"],
            },
        }
        final_path = experiments_dir / "final_test_results.json"
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2)
        print(f"[SAVED] Final test results -> {final_path}")

    print("\n[DONE] Model comparison complete.")
    print(f"  Best model: {best_model_name}  (val macro-F1 = {best_row['macro_f1']:.4f})")
    if not args.run_test:
        print("  To run the test split on the best model, add --run_test")


if __name__ == "__main__":
    main()

