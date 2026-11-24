import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from dataset.dataset_loader import create_datasets, CLASS_NAMES
from models.classification.efficientnet_b0 import (
    create_efficientnet_b0
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 6
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "efficientnet_b0_best.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR / "efficientnet_b0_confusion_matrix.png"
)


# ============================================================
# TRANSFORM
# ============================================================

validation_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DERMAI EFFICIENTNET-B0 TEST EVALUATION")
    print("=" * 60)

    print(f"\nDevice:")
    print(DEVICE)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    _, _, test_dataset = create_datasets(
        train_transform=None,
        validation_transform=validation_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(
        f"Test samples : {len(test_dataset)}"
    )

    print(
        f"Test batches : {len(test_loader)}"
    )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print("\nCreating EfficientNet-B0 model...")

    model = create_efficientnet_b0(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    model = model.to(DEVICE)

    print("Model created successfully.")

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    checkpoint_epoch = checkpoint["epoch"]

    print(
        f"Checkpoint epoch : {checkpoint_epoch}"
    )

    print(
        f"Checkpoint path  : {CHECKPOINT_PATH}"
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    macro_auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro"
    )

    # ========================================================
    # PRINT OVERALL METRICS
    # ========================================================

    print("\n" + "=" * 60)
    print("OVERALL TEST METRICS")
    print("=" * 60)

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Macro Precision   : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall      : {macro_recall:.4f}"
    )

    print(
        f"Macro F1          : {macro_f1:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced_accuracy:.4f}"
    )

    print(
        f"MCC               : {mcc:.4f}"
    )

    print(
        f"Macro AUC-ROC     : {macro_auc:.4f}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=range(NUM_CLASSES),
            target_names=CLASS_NAMES,
            zero_division=0,
            digits=4
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(NUM_CLASSES)
    )

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(cm)

    # ========================================================
    # SENSITIVITY / SPECIFICITY
    # ========================================================

    print("\n" + "=" * 60)
    print("PER-CLASS SENSITIVITY / SPECIFICITY")
    print("=" * 60)

    for i, class_name in enumerate(CLASS_NAMES):

        tp = cm[i, i]

        fn = cm[i, :].sum() - tp

        fp = cm[:, i].sum() - tp

        tn = cm.sum() - (
            tp + fn + fp
        )

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        print(
            f"{class_name}: "
            f"Sensitivity={sensitivity:.4f}, "
            f"Specificity={specificity:.4f}"
        )

    # ========================================================
    # PER-CLASS AUC
    # ========================================================

    print("\n" + "=" * 60)
    print("PER-CLASS AUC-ROC")
    print("=" * 60)

    for i, class_name in enumerate(CLASS_NAMES):

        binary_labels = (
            y_true == i
        ).astype(int)

        class_auc = roc_auc_score(
            binary_labels,
            y_prob[:, i]
        )

        print(
            f"{class_name}: "
            f"AUC={class_auc:.4f}"
        )

    # ========================================================
    # SAVE CONFUSION MATRIX FIGURE
    # ========================================================

    try:

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(
            figsize=(8, 6)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES
        )

        plt.title(
            "EfficientNet-B0 Confusion Matrix"
        )

        plt.xlabel(
            "Predicted Label"
        )

        plt.ylabel(
            "True Label"
        )

        plt.tight_layout()

        plt.savefig(
            CONFUSION_MATRIX_PATH,
            dpi=300
        )

        plt.close()

        print(
            f"\nConfusion matrix saved to:"
        )

        print(
            CONFUSION_MATRIX_PATH
        )

    except Exception as e:

        print(
            "\nCould not save confusion matrix:"
        )

        print(e)

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()