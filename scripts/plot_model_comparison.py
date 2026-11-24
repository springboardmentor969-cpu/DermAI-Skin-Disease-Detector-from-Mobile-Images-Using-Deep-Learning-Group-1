from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL RESULTS
# ============================================================

models = [
    "ResNet-50",
    "DenseNet-121",
    "EfficientNet-B0"
]


accuracy = [
    0.6646,
    0.5884,
    0.6037
]

macro_precision = [
    0.6463,
    0.5100,
    0.5225
]

macro_recall = [
    0.6446,
    0.6054,
    0.5621
]

macro_f1 = [
    0.6298,
    0.5383,
    0.5348
]

balanced_accuracy = [
    0.6446,
    0.6054,
    0.5621
]

mcc = [
    0.5590,
    0.4634,
    0.4756
]

macro_auc = [
    0.8879,
    0.8792,
    0.8755
]


# ============================================================
# GENERAL METRIC COMPARISON
# ============================================================

metric_names = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Balanced Accuracy"
]

metric_values = np.array([
    accuracy,
    macro_precision,
    macro_recall,
    macro_f1,
    balanced_accuracy
])


x = np.arange(len(models))
width = 0.15

plt.figure(
    figsize=(12, 7)
)

for i, metric_name in enumerate(metric_names):

    plt.bar(
        x + (i - 2) * width,
        metric_values[i],
        width,
        label=metric_name
    )

plt.xticks(
    x,
    models
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Model"
)

plt.title(
    "DermAI Milestone 1 CNN Model Comparison"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

comparison_path = (
    OUTPUT_DIR /
    "milestone1_model_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=300
)

plt.close()


# ============================================================
# AUC / MCC COMPARISON
# ============================================================

secondary_metrics = [
    "MCC",
    "Macro AUC-ROC"
]

secondary_values = np.array([
    mcc,
    macro_auc
])

x = np.arange(len(models))
width = 0.3

plt.figure(
    figsize=(10, 6)
)

for i, metric_name in enumerate(secondary_metrics):

    plt.bar(
        x + (i - 0.5) * width,
        secondary_values[i],
        width,
        label=metric_name
    )

plt.xticks(
    x,
    models
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Model"
)

plt.title(
    "DermAI Milestone 1 MCC and AUC-ROC Comparison"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

secondary_path = (
    OUTPUT_DIR /
    "milestone1_mcc_auc_comparison.png"
)

plt.savefig(
    secondary_path,
    dpi=300
)

plt.close()


# ============================================================
# PRINT SUMMARY
# ============================================================

print("=" * 60)
print("DERMAI MILESTONE 1 MODEL COMPARISON")
print("=" * 60)

print()

print(
    f"{'Model':<20}"
    f"{'Accuracy':<12}"
    f"{'Macro F1':<12}"
    f"{'Balanced Acc':<15}"
    f"{'MCC':<10}"
    f"{'AUC':<10}"
)

print("-" * 69)

for i, model in enumerate(models):

    print(
        f"{model:<20}"
        f"{accuracy[i]:<12.4f}"
        f"{macro_f1[i]:<12.4f}"
        f"{balanced_accuracy[i]:<15.4f}"
        f"{mcc[i]:<10.4f}"
        f"{macro_auc[i]:<10.4f}"
    )


# ============================================================
# BEST MODEL
# ============================================================

best_index = np.argmax(macro_f1)

print()

print(
    f"Best model based on Macro F1: "
    f"{models[best_index]}"
)

print()

print(
    "Comparison chart saved to:"
)

print(
    comparison_path
)

print()

print(
    "MCC/AUC chart saved to:"
)

print(
    secondary_path
)

print()

print("=" * 60)
print("COMPARISON COMPLETE")
print("=" * 60)