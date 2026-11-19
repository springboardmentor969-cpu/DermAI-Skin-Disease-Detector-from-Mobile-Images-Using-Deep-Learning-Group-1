"""
DermAI Milestone 2: Model Comparison & Ensemble Recommendation Tool.

Reads metrics.json from every model experiment folder under ml/experiments/,
builds a comparison table across validation metrics, identifies the best model
by validation macro F1, and provides an ensemble recommendation.

Metrics reported:
- model name
- val accuracy
- val macro F1
- val balanced accuracy
- melanoma recall
- ECE (Expected Calibration Error)

Saves comparison to:
- ml/experiments/model_comparison.csv
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="DermAI Model Comparison Tool")
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default=str(PROJECT_ROOT / "ml" / "experiments"),
        help="Path to directory containing experiment folders (default: ml/experiments)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save output CSV (default: <experiments_dir>/model_comparison.csv)",
    )
    return parser.parse_args()


def extract_melanoma_recall(val_data: Dict[str, Any], raw_data: Dict[str, Any]) -> Optional[float]:
    """Extracts melanoma recall (sensitivity) from classification report if available."""
    # 1. Check classification_report in val
    clf_rep = val_data.get("classification_report", {})
    if isinstance(clf_rep, dict):
        if "mel" in clf_rep and isinstance(clf_rep["mel"], dict):
            return float(clf_rep["mel"].get("recall", np.nan))
        if "4" in clf_rep and isinstance(clf_rep["4"], dict):
            return float(clf_rep["4"].get("recall", np.nan))

    # 2. Check per_class breakdown
    per_class = val_data.get("per_class", {})
    if isinstance(per_class, dict) and "mel" in per_class:
        mel_info = per_class["mel"]
        if isinstance(mel_info, dict):
            return float(mel_info.get("recall") or mel_info.get("sensitivity") or np.nan)

    # 3. Check top-level keys
    for k in ["melanoma_recall", "mel_recall"]:
        if k in val_data:
            return float(val_data[k])
        if k in raw_data:
            return float(raw_data[k])

    return np.nan


def extract_ece(val_data: Dict[str, Any], raw_data: Dict[str, Any], exp_dir: Path) -> Optional[float]:
    """Extracts Expected Calibration Error (ECE) from metrics or adjacent calibration files."""
    # 1. In val dict
    if "ece" in val_data and val_data["ece"] is not None:
        try:
            return float(val_data["ece"])
        except (ValueError, TypeError):
            pass

    # 2. In root metrics dict
    if "ece" in raw_data and raw_data["ece"] is not None:
        try:
            return float(raw_data["ece"])
        except (ValueError, TypeError):
            pass

    # 3. Check adjacent calibration summary files in experiment directory
    for candidate_name in ["calibration_results.json", "calibration.json", "val_eval_metrics.json"]:
        cand_file = exp_dir / candidate_name
        if cand_file.is_file():
            try:
                with open(cand_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    if "ece" in cdata:
                        return float(cdata["ece"])
                    if "uncalibrated_ece" in cdata:
                        return float(cdata["uncalibrated_ece"])
                    if "ece_before" in cdata:
                        return float(cdata["ece_before"])
            except Exception:
                pass

    return np.nan


def load_all_experiment_metrics(experiments_dir: Path) -> List[Dict[str, Any]]:
    """Discovers and parses metrics.json from all subdirectories."""
    records = []

    if not experiments_dir.is_dir():
        print(f"Warning: Experiments directory not found: {experiments_dir}")
        return records

    for sub_dir in sorted(experiments_dir.iterdir()):
        if not sub_dir.is_dir():
            continue

        metrics_file = sub_dir / "metrics.json"
        if not metrics_file.is_file():
            continue

        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read {metrics_file}: {e}")
            continue

        model_name = data.get("model") or data.get("model_name") or sub_dir.name
        val_data = data.get("val", {})

        val_acc = val_data.get("accuracy", data.get("accuracy", np.nan))
        val_macro_f1 = val_data.get("macro_f1", data.get("macro_f1", np.nan))
        val_bal_acc = val_data.get("balanced_accuracy", data.get("balanced_accuracy", np.nan))

        mel_recall = extract_melanoma_recall(val_data, data)
        ece = extract_ece(val_data, data, sub_dir)

        # Locate checkpoint
        best_pt = sub_dir / "best.pt"
        last_pt = sub_dir / "last.pt"
        ckpt_path = str(best_pt) if best_pt.is_file() else (str(last_pt) if last_pt.is_file() else str(sub_dir))

        records.append({
            "folder": sub_dir.name,
            "model_name": model_name,
            "val_accuracy": float(val_acc) if val_acc is not None else np.nan,
            "val_macro_f1": float(val_macro_f1) if val_macro_f1 is not None else np.nan,
            "val_balanced_accuracy": float(val_bal_acc) if val_bal_acc is not None else np.nan,
            "melanoma_recall": float(mel_recall) if mel_recall is not None else np.nan,
            "ece": float(ece) if ece is not None and not np.isnan(ece) else np.nan,
            "checkpoint_path": ckpt_path,
        })

    return records


def main():
    args = parse_args()
    exp_dir = Path(args.experiments_dir)
    out_csv = Path(args.output_csv) if args.output_csv else (exp_dir / "model_comparison.csv")

    records = load_all_experiment_metrics(exp_dir)

    if not records:
        print(f"No experiment folders containing metrics.json found under: {exp_dir}")
        return

    # Sort descending by val macro F1
    records.sort(
        key=lambda r: (r["val_macro_f1"] if not np.isnan(r["val_macro_f1"]) else -1.0),
        reverse=True,
    )

    # Build DataFrame
    table_data = []
    for r in records:
        table_data.append({
            "model_name": r["model_name"],
            "val_accuracy": r["val_accuracy"],
            "val_macro_f1": r["val_macro_f1"],
            "val_balanced_accuracy": r["val_balanced_accuracy"],
            "melanoma_recall": r["melanoma_recall"],
            "ece": r["ece"],
        })

    df = pd.DataFrame(table_data)

    # Save to CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Print Table to stdout
    print("=" * 88)
    print(f"{'DermAI Milestone 2 — Model Comparison Table':^88}")
    print("=" * 88)

    header = f"{'Model Name':<28} {'Val Accuracy':<14} {'Val Macro F1':<14} {'Val Bal Acc':<14} {'Mel Recall':<12} {'ECE':<8}"
    print(header)
    print("-" * 88)

    for r in records:
        m_name = r["model_name"]
        acc_str = f"{r['val_accuracy']:.4f}" if not np.isnan(r["val_accuracy"]) else "N/A"
        f1_str = f"{r['val_macro_f1']:.4f}" if not np.isnan(r["val_macro_f1"]) else "N/A"
        bal_str = f"{r['val_balanced_accuracy']:.4f}" if not np.isnan(r["val_balanced_accuracy"]) else "N/A"
        mel_str = f"{r['melanoma_recall']:.4f}" if not np.isnan(r["melanoma_recall"]) else "N/A"
        ece_str = f"{r['ece']:.4f}" if not np.isnan(r["ece"]) else "N/A"

        print(f"{m_name:<28} {acc_str:<14} {f1_str:<14} {bal_str:<14} {mel_str:<12} {ece_str:<8}")

    print("-" * 88)
    print(f"Table saved to: {out_csv}")

    # Best model
    best = records[0]
    best_f1_str = f"{best['val_macro_f1']:.4f}" if not np.isnan(best["val_macro_f1"]) else "N/A"
    print(f"\nBest model by val macro F1: {best['model_name']} (val macro F1: {best_f1_str})")

    # Ensemble recommendation
    if len(records) >= 2:
        top1_path = records[0]["checkpoint_path"]
        top2_path = records[1]["checkpoint_path"]
        print(f"Ensemble recommendation: Combine top 2 models by val macro F1 — paths: {top1_path}, {top2_path}")
    else:
        top1_path = records[0]["checkpoint_path"]
        print(f"Ensemble recommendation: Combine top 2 models by val macro F1 — paths: {top1_path}")


if __name__ == "__main__":
    main()

