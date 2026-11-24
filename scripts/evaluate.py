import sys
from pathlib import Path

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
from training.training_config import BEST_MODEL_PATH


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DERMAI RESNET-50 TEST EVALUATION")
print("=" * 60)

print("\nDevice:")
print(DEVICE)


# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

class_names = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK"
]


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

print(f"Test samples : {len(test_dataset)}")
print(f"Test batches : {len(test_loader)}")


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

print("\nCreating ResNet-50 model...")

model = create_resnet50(
    num_classes=6,
    pretrained=False
)

model = model.to(DEVICE)

print("Model created successfully.")


# --------------------------------------------------
# LOAD BEST CHECKPOINT
# --------------------------------------------------

print("\nLoading best checkpoint...")
print(BEST_MODEL_PATH)

checkpoint = torch.load(
    BEST_MODEL_PATH,
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


# --------------------------------------------------
# TEST EVALUATION
# --------------------------------------------------

print("\nRunning test evaluation...")

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Model output
        outputs = model(images)

        # Convert logits to probabilities
        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Select class with highest probability
        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # Store actual labels
        all_labels.extend(
            labels.cpu().numpy()
        )

        # Store predicted labels
        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Store probabilities for AUC-ROC
        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


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


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# --------------------------------------------------
# MATTHEWS CORRELATION COEFFICIENT
# --------------------------------------------------

mcc = matthews_corrcoef(
    all_labels,
    all_predictions
)


# --------------------------------------------------
# AUC-ROC
# --------------------------------------------------

all_labels_array = np.array(
    all_labels
)

all_probabilities_array = np.array(
    all_probabilities
)

# Convert class labels into one-hot format
one_hot_labels = np.eye(
    len(class_names)
)[all_labels_array]

# One-vs-Rest AUC for every class
auc_per_class = roc_auc_score(
    one_hot_labels,
    all_probabilities_array,
    multi_class="ovr",
    average=None
)

# Macro AUC across all classes
macro_auc = roc_auc_score(
    one_hot_labels,
    all_probabilities_array,
    multi_class="ovr",
    average="macro"
)


# --------------------------------------------------
# TEST RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("TEST RESULTS")
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


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

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


# --------------------------------------------------
# PER-CLASS SENSITIVITY AND SPECIFICITY
# --------------------------------------------------

print("=" * 60)
print("PER-CLASS SENSITIVITY AND SPECIFICITY")
print("=" * 60)

for i, class_name in enumerate(class_names):

    # True Positive
    true_positive = cm[i, i]

    # False Negative
    false_negative = (
        cm[i, :].sum() - true_positive
    )

    # False Positive
    false_positive = (
        cm[:, i].sum() - true_positive
    )

    # True Negative
    true_negative = cm.sum() - (
        true_positive
        + false_negative
        + false_positive
    )

    # Sensitivity
    sensitivity = (
        true_positive /
        (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    # Specificity
    specificity = (
        true_negative /
        (true_negative + false_positive)
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
    class_names,
    auc_per_class
):

    print(
        f"{class_name}: "
        f"AUC-ROC = {auc:.4f}"
    )


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(cm)


# --------------------------------------------------
# FINISHED
# --------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)