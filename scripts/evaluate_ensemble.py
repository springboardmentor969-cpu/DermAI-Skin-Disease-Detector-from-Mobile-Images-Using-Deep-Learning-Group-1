
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

from backend.services.inference import (
    load_model,
    DEVICE,
    CLASS_NAMES
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

BATCH_SIZE = 16
NUM_CLASSES = 6

CHECKPOINT_PATH = (
    ROOT_DIR
    / "models"
    / "checkpoints"
    / "vit_b16_best.pth"
)

OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CM_PATH = OUTPUT_DIR / "ensemble_confusion_matrix.png"
REPORT_PATH = OUTPUT_DIR / "ensemble_classification_report.txt"

CLASS_NAMES = [
    "ACK", "BCC", "MEL", "NEV", "SCC", "SEK"
]


# ============================================================
# LOAD VIT MODEL
# ============================================================

def load_vit_model():

    print("\nLoading ViT-B/16...")

    model = create_vit(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    print("ViT-B/16 loaded successfully.")
    print("Best epoch:", checkpoint["epoch"])

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DERMAI THREE-MODEL SOFT-VOTING ENSEMBLE")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"ViT checkpoint not found: {CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # TEST DATASET
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    _, _, test_dataset = create_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Test samples:", len(test_dataset))
    print("Test batches:", len(test_loader))

    # --------------------------------------------------------
    # LOAD THREE TRAINED MODELS
    # --------------------------------------------------------

    print("\nLoading trained models...")

    resnet_model = load_model("resnet50")
    densenet_model = load_model("densenet121")
    vit_model = load_vit_model()

    resnet_model.eval()
    densenet_model.eval()
    vit_model.eval()

    print("\nAll three models loaded successfully.")

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    y_true = []

    resnet_probs = []
    densenet_probs = []
    vit_probs = []
    ensemble_probs = []

    print("\nRunning ensemble evaluation...")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            # ResNet-50
            resnet_output = resnet_model(images)

            # DenseNet-121
            densenet_output = densenet_model(images)

            # ViT-B/16
            vit_output = vit_model(images)

            # Convert logits to probabilities
            p_resnet = torch.softmax(
                resnet_output,
                dim=1
            )

            p_densenet = torch.softmax(
                densenet_output,
                dim=1
            )

            p_vit = torch.softmax(
                vit_output,
                dim=1
            )

            # Equal-weight soft voting
            p_ensemble = (
                p_resnet
                + p_densenet
                + p_vit
            ) / 3.0

            y_true.extend(labels.numpy())

            resnet_probs.extend(
                p_resnet.cpu().numpy()
            )

            densenet_probs.extend(
                p_densenet.cpu().numpy()
            )

            vit_probs.extend(
                p_vit.cpu().numpy()
            )

            ensemble_probs.extend(
                p_ensemble.cpu().numpy()
            )

    y_true = np.asarray(y_true)

    # --------------------------------------------------------
    # METRIC FUNCTION
    # --------------------------------------------------------

    def calculate_metrics(probabilities):

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        return {
            "accuracy": accuracy_score(
                y_true, predictions
            ),
            "precision": precision_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0
            ),
            "recall": recall_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0
            ),
            "f1": f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_true, predictions
            ),
            "mcc": matthews_corrcoef(
                y_true, predictions
            ),
            "auc": roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=np.arange(NUM_CLASSES)
            )
        }

    # --------------------------------------------------------
    # COMPARE ALL MODELS
    # --------------------------------------------------------

    model_probabilities = {
        "ResNet-50": np.asarray(resnet_probs),
        "DenseNet-121": np.asarray(densenet_probs),
        "ViT-B/16": np.asarray(vit_probs),
        "Ensemble": np.asarray(ensemble_probs)
    }

    results = {}

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    for name, probabilities in model_probabilities.items():

        metrics = calculate_metrics(probabilities)
        results[name] = metrics

        print(f"\n{name}")

        for metric, value in metrics.items():
            print(f"{metric:22s}: {value:.4f}")

    # --------------------------------------------------------
    # ENSEMBLE CLASSIFICATION REPORT
    # --------------------------------------------------------

    y_pred = np.argmax(
        ensemble_probs,
        axis=1
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print("\nENSEMBLE CLASSIFICATION REPORT")
    print("-" * 70)
    print(report)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES)
    )

    print("\nENSEMBLE CONFUSION MATRIX")
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

    plt.title("DermAI Three-Model Ensemble Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(
        CM_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("DERMAI THREE-MODEL ENSEMBLE EVALUATION\n\n")

        for name, metrics in results.items():

            file.write(f"\n{name}\n")

            for metric, value in metrics.items():
                file.write(
                    f"{metric}: {value:.4f}\n"
                )

        file.write("\nENSEMBLE CLASSIFICATION REPORT\n")
        file.write(report)

        file.write("\nENSEMBLE CONFUSION MATRIX\n")
        file.write(str(cm))

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\nSaved confusion matrix:")
    print(CM_PATH)

    print("\nSaved evaluation report:")
    print(REPORT_PATH)

    print("\nEnsemble evaluation completed.")

    print("=" * 70)


if __name__ == "__main__":
    main()
