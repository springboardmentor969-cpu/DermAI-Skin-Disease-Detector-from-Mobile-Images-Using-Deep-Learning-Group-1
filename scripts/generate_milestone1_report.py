from pathlib import Path
import json


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "milestone1_results.json"


# ============================================================
# DATASET SUMMARY
# ============================================================

dataset_summary = {
    "dataset": "PAD-UFES-20",

    "total_images": 2298,
    "total_patients": 1373,
    "total_lesions": 1641,

    "image_size": "224x224",
    "image_channels": 3,

    "classes": [
        "ACK",
        "BCC",
        "MEL",
        "NEV",
        "SCC",
        "SEK"
    ]
}


# ============================================================
# PATIENT-LEVEL SPLIT
# ============================================================

split_summary = {
    "train": {
        "images": 1634,
        "patients": 961
    },

    "validation": {
        "images": 336,
        "patients": 206
    },

    "test": {
        "images": 328,
        "patients": 206
    },

    "patient_overlap": {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0
    }
}


# ============================================================
# MODEL RESULTS
# ============================================================

model_results = {

    "ResNet-50": {

        "accuracy": 0.6646,
        "macro_precision": 0.6463,
        "macro_recall": 0.6446,
        "macro_f1": 0.6298,
        "balanced_accuracy": 0.6446,
        "mcc": 0.5590,
        "macro_auc_roc": 0.8879,

        "best_checkpoint_epoch": 5,

        "per_class": {
            "ACK": {
                "precision": 0.72,
                "recall": 0.73,
                "f1": 0.73
            },

            "BCC": {
                "precision": 0.76,
                "recall": 0.62,
                "f1": 0.69
            },

            "MEL": {
                "precision": 0.80,
                "recall": 0.50,
                "f1": 0.62
            },

            "NEV": {
                "precision": 0.65,
                "recall": 0.71,
                "f1": 0.68
            },

            "SCC": {
                "precision": 0.45,
                "recall": 0.44,
                "f1": 0.45
            },

            "SEK": {
                "precision": 0.49,
                "recall": 0.86,
                "f1": 0.62
            }
        }
    },


    "DenseNet-121": {

        "accuracy": 0.5884,
        "macro_precision": 0.5100,
        "macro_recall": 0.6054,
        "macro_f1": 0.5383,
        "balanced_accuracy": 0.6054,
        "mcc": 0.4634,
        "macro_auc_roc": 0.8792,

        "best_checkpoint_epoch": 3,

        "per_class": {
            "ACK": {
                "precision": 0.73,
                "recall": 0.67,
                "f1": 0.70
            },

            "BCC": {
                "precision": 0.71,
                "recall": 0.58,
                "f1": 0.64
            },

            "MEL": {
                "precision": 0.41,
                "recall": 0.88,
                "f1": 0.56
            },

            "NEV": {
                "precision": 0.57,
                "recall": 0.62,
                "f1": 0.59
            },

            "SCC": {
                "precision": 0.21,
                "recall": 0.24,
                "f1": 0.22
            },

            "SEK": {
                "precision": 0.43,
                "recall": 0.66,
                "f1": 0.52
            }
        }
    },


    "EfficientNet-B0": {

        "accuracy": 0.6037,
        "macro_precision": 0.5225,
        "macro_recall": 0.5621,
        "macro_f1": 0.5348,
        "balanced_accuracy": 0.5621,
        "mcc": 0.4756,
        "macro_auc_roc": 0.8755,

        "best_checkpoint_epoch": 5,

        "per_class": {
            "ACK": {
                "precision": 0.66,
                "recall": 0.73,
                "f1": 0.70
            },

            "BCC": {
                "precision": 0.72,
                "recall": 0.58,
                "f1": 0.64
            },

            "MEL": {
                "precision": 0.33,
                "recall": 0.50,
                "f1": 0.40
            },

            "NEV": {
                "precision": 0.75,
                "recall": 0.71,
                "f1": 0.73
            },

            "SCC": {
                "precision": 0.24,
                "recall": 0.24,
                "f1": 0.24
            },

            "SEK": {
                "precision": 0.43,
                "recall": 0.62,
                "f1": 0.51
            }
        }
    }
}


# ============================================================
# BEST MODEL
# ============================================================

best_model = max(
    model_results,
    key=lambda model: model_results[model]["macro_f1"]
)


# ============================================================
# IMPORTANT OBSERVATIONS
# ============================================================

observations = [

    "ResNet-50 achieved the strongest overall baseline performance.",

    "ResNet-50 achieved the highest accuracy, macro precision, macro F1, "
    "balanced accuracy, MCC, and macro AUC-ROC among the three models.",

    "SCC classification is a major weakness across all three baseline models.",

    "MEL test performance should be interpreted cautiously because the "
    "test set contains only 8 MEL samples.",

    "Patient-level splitting prevented patient overlap between train, "
    "validation, and test sets.",

    "The baseline results are research results and should not be interpreted "
    "as clinical-ready diagnostic performance."
]


# ============================================================
# FINAL REPORT
# ============================================================

report = {

    "project": "DermAI",

    "milestone": "Milestone 1",

    "title": "Dataset Preparation and CNN Baseline Evaluation",

    "dataset_summary": dataset_summary,

    "split_summary": split_summary,

    "model_results": model_results,

    "best_model": best_model,

    "selection_metric": "Macro F1",

    "observations": observations
}


# ============================================================
# SAVE JSON
# ============================================================

with open(REPORT_PATH, "w", encoding="utf-8") as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("=" * 70)
print("DERMAI MILESTONE 1 FINAL REPORT")
print("=" * 70)

print()

print("Dataset:")
print(f"  Dataset       : {dataset_summary['dataset']}")
print(f"  Images        : {dataset_summary['total_images']}")
print(f"  Patients      : {dataset_summary['total_patients']}")
print(f"  Lesions       : {dataset_summary['total_lesions']}")

print()

print("Patient-Level Split:")
print(
    f"  Train         : "
    f"{split_summary['train']['images']} images / "
    f"{split_summary['train']['patients']} patients"
)

print(
    f"  Validation    : "
    f"{split_summary['validation']['images']} images / "
    f"{split_summary['validation']['patients']} patients"
)

print(
    f"  Test          : "
    f"{split_summary['test']['images']} images / "
    f"{split_summary['test']['patients']} patients"
)

print()

print("Model Performance:")
print("-" * 70)

print(
    f"{'Model':<20}"
    f"{'Accuracy':<12}"
    f"{'Macro F1':<12}"
    f"{'Balanced Acc':<15}"
    f"{'MCC':<10}"
    f"{'AUC':<10}"
)

print("-" * 70)

for model_name, results in model_results.items():

    print(
        f"{model_name:<20}"
        f"{results['accuracy']:<12.4f}"
        f"{results['macro_f1']:<12.4f}"
        f"{results['balanced_accuracy']:<15.4f}"
        f"{results['mcc']:<10.4f}"
        f"{results['macro_auc_roc']:<10.4f}"
    )

print()

print(f"Selected baseline model : {best_model}")
print(f"Selection metric        : Macro F1")
print(f"Macro F1                : "
      f"{model_results[best_model]['macro_f1']:.4f}")

print()

print("Important observations:")

for observation in observations:

    print(f"  - {observation}")

print()

print("Report saved to:")
print(REPORT_PATH)

print()

print("=" * 70)
print("MILESTONE 1 REPORT GENERATED SUCCESSFULLY")
print("=" * 70)