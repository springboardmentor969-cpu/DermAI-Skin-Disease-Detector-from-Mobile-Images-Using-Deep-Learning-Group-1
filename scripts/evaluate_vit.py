
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from dataset.dataset_loader import create_datasets
from models.classification.vit import create_vit


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
NUM_CLASSES = 6

CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK"
]

ROOT_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    ROOT_DIR / "models" / "checkpoints" / "vit_b16_best.pth"
)

OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR / "vit_b16_confusion_matrix.png"
)

REPORT_PATH = (
    OUTPUT_DIR / "vit_b16_classification_report.txt"
)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 65)
    print("DERMAI VISION TRANSFORMER (ViT-B/16) TEST EVALUATION")
    print("=" * 65)

    print("\nDevice:")
    print(DEVICE)

    # --------------------------------------------------------
    # LOAD TEST DATASET
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    train_dataset, val_dataset, test_dataset = create_datasets()

    print("Test samples :", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Test batches :", len(test_loader))

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print("\nCreating ViT-B/16 model...")

    model = create_vit(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    model = model.to(DEVICE)

    print("Model created successfully.")

    # --------------------------------------------------------
    # LOAD BEST CHECKPOINT
    # --------------------------------------------------------

    print("\nLoading best checkpoint...")
    print(CHECKPOINT_PATH)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Checkpoint epoch :", checkpoint["epoch"])
    print("Validation loss  :", f"{checkpoint['val_loss']:.4f}")
    print("Validation acc   :", f"{checkpoint['val_accuracy']:.4f}")

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    print("\nRunning test evaluation...")

    model.eval()

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
                probabilities,
                dim=1
            )

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    try:
        auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=np.arange(NUM_CLASSES)
        )
    except ValueError:
        auc = float("nan")
        print("\nWarning: Macro AUC-ROC could not be calculated.")

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("ViT-B/16 TEST RESULTS")
    print("=" * 65)

    print("Test samples       :", len(y_true))
    print("Accuracy           :", f"{accuracy:.4f}")
    print("Macro Precision    :", f"{precision:.4f}")
    print("Macro Recall       :", f"{recall:.4f}")
    print("Macro F1-score     :", f"{f1:.4f}")
    print("Balanced Accuracy  :", f"{balanced_acc:.4f}")
    print("MCC                :", f"{mcc:.4f}")
    print("Macro AUC-ROC      :", f"{auc:.4f}")

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT")
    print("-" * 65)
    print(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("DERMAI ViT-B/16 TEST EVALUATION\n\n")
        file.write(f"Test samples: {len(y_true)}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro Precision: {precision:.4f}\n")
        file.write(f"Macro Recall: {recall:.4f}\n")
        file.write(f"Macro F1-score: {f1:.4f}\n")
        file.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        file.write(f"MCC: {mcc:.4f}\n")
        file.write(f"Macro AUC-ROC: {auc:.4f}\n\n")
        file.write(report)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES)
    )

    print("CONFUSION MATRIX")
    print("-" * 65)
    print(cm)

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.title("DermAI ViT-B/16 Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\nSaved classification report:")
    print(REPORT_PATH)

    print("\nSaved confusion matrix:")
    print(CONFUSION_MATRIX_PATH)

    print("\nEvaluation completed successfully.")

    print("=" * 65)


if __name__ == "__main__":
    main()
