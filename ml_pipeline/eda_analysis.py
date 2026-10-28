#!/usr/bin/env python3
'''
DermAI Milestone 1 — Exploratory Data Analysis (EDA) & Class Imbalance Analyzer.
Calculates class percentages, imbalance ratios, and balanced class weights
for weighted Cross-Entropy and Focal Loss training in Milestone 2.
'''
import argparse
import json
from dataset_config import DATASETS

def compute_eda_summary(dataset_key: str) -> dict:
    ds = DATASETS[dataset_key]
    total = ds["total_images"]
    num_classes = len(ds["classes"])
    counts = [c["count"] for c in ds["classes"]]
    max_count = max(counts)
    min_count = min(counts)

    enriched_classes = []
    for c in ds["classes"]:
        cnt = c["count"]
        pct = round((cnt / total) * 100.0, 2)
        # Balanced inverse-frequency class weight: N / (K * n_c)
        balanced_weight = round(total / (num_classes * cnt), 4)
        enriched_classes.append({
            **c,
            "percentage": pct,
            "imbalance_ratio_vs_majority": round(max_count / cnt, 2),
            "computed_class_weight": balanced_weight,
        })

    return {
        "dataset_id": dataset_key,
        "name": ds["name"],
        "modality": ds["modality"],
        "total_images": total,
        "num_classes": num_classes,
        "max_to_min_imbalance_ratio": round(max_count / min_count, 2),
        "recommended_split": {
            "train_80pct": int(total * 0.80),
            "val_10pct": int(total * 0.10),
            "test_10pct": total - int(total * 0.80) - int(total * 0.10),
        },
        "class_distribution": enriched_classes,
        "anatomical_sites": ds["anatomical_sites"],
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DermAI Milestone 1 Dataset EDA")
    parser.add_argument("--dataset", default="ham10000", choices=list(DATASETS.keys()))
    parser.add_argument("--output", default="eda_summary.json")
    args = parser.parse_args()

    report = compute_eda_summary(args.dataset)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[EDA Complete] Saved {args.dataset} analysis ({report['total_images']} images, {report['num_classes']} classes) -> {args.output}")
