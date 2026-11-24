import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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

from models.classification.densenet121 import create_densenet121
from dataset.dataset_loader import create_datasets
from training.data_transforms import validation_transform
from training.training_config import CHECKPOINT_DIR


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DENSENET_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "densenet121_best.pth"
)

class_names = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK"
]


print("=" * 60)
print("DERMAI DENSENET-121 TEST EVALUATION")
print("=" * 60)

print("\nDevice:")
print(DEVICE)


# ------------------------------------------------------------
# TEST DATASET
# ------------------------------------------------------------

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

print(f"Test samples : {len(test_dataset)}")
print(f"Test batches : {len(test_loader)}")


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

print("\nCreating DenseNet-121 model...")

model = create_densenet121(
    num_classes=6,
    pretrained=False
)

model = model.to(DEVICE)

print("Model created successfully.")


# ------------------------------------------------------------
# LOAD CHECKPOINT
# ------------------------------------------------------------

print("\nLoading best checkpoint...")
print(DENSENET_CHECKPOINT_PATH)

checkpoint = torch.load(
    DENSENET_CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Checkpoint epoch : "
    f"{checkpoint['epoch']}"
)

print(
    f"Validation loss  : "
    f"{checkpoint['validation_loss']:.4f}"
)

print(
    f"Validation acc   : "
    f"{checkpoint['validation_accuracy']:.4f}"
)

model.eval()


# ------------------------------------------------------------
# TEST PREDICTIONS
# ------------------------------------------------------------

print("\nRunning test evaluation...")

all_labels = []
all_predictions = []
all_probabilities = []


with torch.no_grad():

    for images, labels in test_loader:

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


# ------------------------------------------------------------
# BASIC METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
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

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ------------------------------------------------------------
# AUC-ROC
# ------------------------------------------------------------

all_labels_array = np.array(
    all_labels
)

all_probabilities_array = np.array(
    all_probabilities
)

one_hot_labels = np.eye(
    len(class_names)
)[all_labels_array]


auc_per_class = roc_auc_score(
    one_hot_labels,
    all_probabilities_array,
    multi_class="ovr",
    average=None
)

macro_auc = roc_auc_score(
    one_hot_labels,
    all_probabilities_array,
    multi_class="ovr",
    average="macro"
)


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DENSENET-121 TEST RESULTS")
print("=" * 60)

print(
    f"\nAccuracy           : "
    f"{accuracy:.4f}"
)

print(
    f"Macro Precision    : "
    f"{precision:.4f}"
)

print(
    f"Macro Recall       : "
    f"{recall:.4f}"
)

print(
    f"Macro F1-score     : "
    f"{f1:.4f}"
)

print(
    f"Balanced Accuracy  : "
    f"{balanced_accuracy:.4f}"
)

print(
    f"MCC                : "
    f"{mcc:.4f}"
)

print(
    f"Macro AUC-ROC      : "
    f"{macro_auc:.4f}"
)


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)


# ------------------------------------------------------------
# SENSITIVITY AND SPECIFICITY
# ------------------------------------------------------------

print("=" * 60)
print("PER-CLASS SENSITIVITY AND SPECIFICITY")
print("=" * 60)


for i, class_name in enumerate(class_names):

    true_positive = cm[i, i]

    false_negative = (
        cm[i, :].sum()
        - true_positive
    )

    false_positive = (
        cm[:, i].sum()
        - true_positive
    )

    true_negative = (
        cm.sum()
        - (
            true_positive
            + false_negative
            + false_positive
        )
    )

    sensitivity = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    specificity = (
        true_negative
        / (true_negative + false_positive)
        if (true_negative + false_positive) > 0
        else 0
    )

    print(
        f"{class_name}: "
        f"Sensitivity = {sensitivity:.4f}, "
        f"Specificity = {specificity:.4f}"
    )


# ------------------------------------------------------------
# PER-CLASS AUC
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PER-CLASS AUC-ROC")
print("=" * 60)


for class_name, auc in zip(
    class_names,
    auc_per_class
):

    print(
        f"{class_name}: "
        f"AUC-ROC = {auc:.4f}"
    )


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(cm)


print("\n" + "=" * 60)
print("DENSENET-121 EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)