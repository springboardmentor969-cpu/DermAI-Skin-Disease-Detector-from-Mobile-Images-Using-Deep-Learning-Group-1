
import sys
import csv
from pathlib import Path

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    classification_report,
    matthews_corrcoef,
    roc_auc_score
)

from torch.utils.data import DataLoader

from models.classification.resnet50 import create_resnet50
from dataset.dataset_loader import create_datasets
from training.data_transforms import validation_transform


# --------------------------------------------------
# DEVICE AND PATHS
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TUNED_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "resnet50_tuned_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tuned_resnet50"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK"
]

NUM_CLASSES = len(CLASS_NAMES)
CLASS_LABELS = np.arange(NUM_CLASSES)


# --------------------------------------------------
# START
# --------------------------------------------------

print("=" * 60)
print("DERMAI TUNED RESNET-50 TEST EVALUATION")
print("=" * 60)

print("\nDevice:")
print(DEVICE)


# --------------------------------------------------
# CHECK CHECKPOINT
# --------------------------------------------------

if not TUNED_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Tuned checkpoint not found: {TUNED_MODEL_PATH}"
    )


# --------------------------------------------------
# LOAD TEST DATASET
# --------------------------------------------------

print("\nLoading test dataset...")

_, _, test_dataset = create_datasets(
    validation_transform=validation_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))
print("Test batches:", len(test_loader))


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

print("\nCreating ResNet-50 model...")

model = create_resnet50(
    num_classes=NUM_CLASSES,
    pretrained=False
)

model = model.to(DEVICE)

print("Model created successfully.")


# --------------------------------------------------
# LOAD TUNED CHECKPOINT
# --------------------------------------------------

print("\nLoading tuned checkpoint...")
print(TUNED_MODEL_PATH)

checkpoint = torch.load(
    TUNED_MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("Checkpoint epoch:", checkpoint["epoch"])
print(
    f"Validation loss: "
    f"{checkpoint['validation_loss']:.4f}"
)
print(
    f"Validation accuracy: "
    f"{checkpoint['validation_accuracy']:.4f}"
)
print(
    f"Validation macro F1: "
    f"{checkpoint['validation_macro_f1']:.4f}"
)
print(
    f"Validation balanced accuracy: "
    f"{checkpoint['validation_balanced_accuracy']:.4f}"
)

print("Learning rate:", checkpoint["learning_rate"])
print("Weight decay:", checkpoint["weight_decay"])

model.eval()


# --------------------------------------------------
# TEST EVALUATION
# --------------------------------------------------

print("\nRunning test evaluation...")

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        if (batch_index + 1) % 5 == 0:
            print(
                f"Processed batch "
                f"{batch_index + 1}/{len(test_loader)}"
            )


# --------------------------------------------------
# CONVERT TO NUMPY
# --------------------------------------------------

all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)


# --------------------------------------------------
# BASIC METRICS
# --------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    labels=CLASS_LABELS,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    labels=CLASS_LABELS,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    labels=CLASS_LABELS,
    average="macro",
    zero_division=0
)

balanced_accuracy = balanced_accuracy_score(
    all_labels,
    all_predictions
)

mcc = matthews_corrcoef(
    all_labels,
    all_predictions
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=CLASS_LABELS
)


# --------------------------------------------------
# MULTICLASS AUC-ROC
# --------------------------------------------------

auc_per_class = roc_auc_score(
    all_labels,
    all_probabilities,
    labels=CLASS_LABELS,
    multi_class="ovr",
    average=None
)

macro_auc = roc_auc_score(
    all_labels,
    all_probabilities,
    labels=CLASS_LABELS,
    multi_class="ovr",
    average="macro"
)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

report = classification_report(
    all_labels,
    all_predictions,
    labels=CLASS_LABELS,
    target_names=CLASS_NAMES,
    zero_division=0
)


# --------------------------------------------------
# PRINT TEST RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("TUNED RESNET-50 TEST RESULTS")
print("=" * 60)

print(f"\nAccuracy           : {accuracy:.4f}")
print(f"Macro Precision    : {precision:.4f}")
print(f"Macro Recall       : {recall:.4f}")
print(f"Macro F1-score     : {f1:.4f}")
print(f"Balanced Accuracy  : {balanced_accuracy:.4f}")
print(f"MCC                : {mcc:.4f}")
print(f"Macro AUC-ROC      : {macro_auc:.4f}")


# --------------------------------------------------
# PRINT CLASSIFICATION REPORT
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)


# --------------------------------------------------
# SENSITIVITY AND SPECIFICITY
# --------------------------------------------------

print("=" * 60)
print("PER-CLASS SENSITIVITY AND SPECIFICITY")
print("=" * 60)

for i, class_name in enumerate(CLASS_NAMES):

    true_positive = cm[i, i]

    false_negative = (
        cm[i, :].sum() - true_positive
    )

    false_positive = (
        cm[:, i].sum() - true_positive
    )

    true_negative = (
        cm.sum()
        - true_positive
        - false_negative
        - false_positive
    )

    sensitivity = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    specificity = (
        true_negative / (true_negative + false_positive)
        if (true_negative + false_positive) > 0
        else 0
    )

    print(
        f"{class_name}: "
        f"Sensitivity = {sensitivity:.4f}, "
        f"Specificity = {specificity:.4f}"
    )


# --------------------------------------------------
# PER-CLASS AUC-ROC
# --------------------------------------------------

print("\n" + "=" * 60)
print("PER-CLASS AUC-ROC")
print("=" * 60)

for class_name, auc in zip(
    CLASS_NAMES,
    auc_per_class
):

    print(
        f"{class_name}: AUC-ROC = {auc:.4f}"
    )


# --------------------------------------------------
# PRINT CONFUSION MATRIX
# --------------------------------------------------

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("Rows = Actual")
print("Columns = Predicted\n")

print(cm)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

metrics = {
    "accuracy": accuracy,
    "macro_precision": precision,
    "macro_recall": recall,
    "macro_f1": f1,
    "balanced_accuracy": balanced_accuracy,
    "mcc": mcc,
    "macro_auc_roc": macro_auc
}

metrics_path = OUTPUT_DIR / "tuned_resnet50_metrics.csv"

with open(
    metrics_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])

    for name, value in metrics.items():
        writer.writerow([name, f"{value:.6f}"])


report_path = OUTPUT_DIR / "tuned_resnet50_classification_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


cm_path = OUTPUT_DIR / "tuned_resnet50_confusion_matrix.csv"

np.savetxt(
    cm_path,
    cm,
    delimiter=",",
    fmt="%d"
)


# --------------------------------------------------
# FINISHED
# --------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nSaved outputs:")
print(metrics_path)
print(report_path)
print(cm_path)