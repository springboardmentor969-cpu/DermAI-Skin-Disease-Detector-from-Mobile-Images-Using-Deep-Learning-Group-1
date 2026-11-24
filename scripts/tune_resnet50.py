
from pathlib import Path
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset.dataset_loader import create_datasets
from models.classification.resnet50 import create_resnet50


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.0001
NUM_CLASSES = 6

LEARNING_RATES = [
    0.00005,
    0.0001,
    0.0002
]

CLASS_NAMES = [
    "ACK", "BCC", "MEL", "NEV", "SCC", "SEK"
]

CLASS_WEIGHTS = torch.tensor(
    [
        0.5177,
        0.4592,
        7.5648,
        1.5473,
        2.0323,
        1.6114
    ],
    dtype=torch.float32
).to(DEVICE)

OUTPUT_DIR = ROOT_DIR / "outputs" / "hyperparameter_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = (
    ROOT_DIR
    / "models"
    / "checkpoints"
    / "resnet50_tuned_best.pth"
)

RESULTS_CSV = OUTPUT_DIR / "resnet50_tuning_results.csv"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ============================================================
# VALIDATION
# ============================================================

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# ============================================================
# RUN ONE HYPERPARAMETER TRIAL
# ============================================================

def run_trial(learning_rate, train_loader, val_loader):

    print("\n" + "=" * 65)
    print(f"TRIAL: Learning Rate = {learning_rate}")
    print("=" * 65)

    set_seed(42)

    model = create_resnet50(
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch

    print("\nTrial completed.")
    print("Learning rate:", learning_rate)
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 4))
    print("Best validation accuracy:", round(best_val_accuracy, 4))

    return {
        "learning_rate": learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("DERMAI RESNET-50 HYPERPARAMETER OPTIMIZATION")
    print("=" * 65)

    print("Device:", DEVICE)
    print("Epochs per trial:", NUM_EPOCHS)
    print("Batch size:", BATCH_SIZE)

    print("\nLoading datasets...")

    train_dataset, val_dataset, _ = create_datasets()

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    results = []

    for learning_rate in LEARNING_RATES:

        result = run_trial(
            learning_rate,
            train_loader,
            val_loader
        )

        results.append(result)

    # --------------------------------------------------------
    # SELECT BEST TRIAL
    # --------------------------------------------------------

    best_result = min(
        results,
        key=lambda item: item["best_val_loss"]
    )

    print("\n" + "=" * 65)
    print("HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("=" * 65)

    for result in results:

        print(
            f"LR: {result['learning_rate']:.6f} | "
            f"Best Epoch: {result['best_epoch']} | "
            f"Val Loss: {result['best_val_loss']:.4f} | "
            f"Val Acc: {result['best_val_accuracy']:.4f}"
        )

    print("\nBest learning rate:", best_result["learning_rate"])
    print("Best weight decay:", best_result["weight_decay"])
    print("Best validation loss:", best_result["best_val_loss"])
    print("Best validation accuracy:", best_result["best_val_accuracy"])

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved:")
    print(RESULTS_CSV)

    print("\nNote:")
    print(
        "The selected hyperparameters are identified using "
        "validation loss only."
    )
    print(
        "Run a separate final training with the selected "
        "configuration before evaluating on the test set."
    )

    print("\nOptimization completed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
