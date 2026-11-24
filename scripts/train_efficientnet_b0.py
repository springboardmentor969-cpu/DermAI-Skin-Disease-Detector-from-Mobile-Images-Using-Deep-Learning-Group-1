import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from dataset.dataset_loader import create_datasets
from models.classification.efficientnet_b0 import create_efficientnet_b0


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 10

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT / "models" / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR /
    "efficientnet_b0_best.pth"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

# Class order:
# ACK, BCC, MEL, NEV, SCC, SEK

CLASS_WEIGHTS = torch.tensor(
    [
        0.5177,   # ACK
        0.4592,   # BCC
        7.5648,   # MEL
        1.5473,   # NEV
        2.0323,   # SCC
        1.6114    # SEK
    ],
    dtype=torch.float32
).to(DEVICE)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


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
    print("DERMAI EFFICIENTNET-B0 TRAINING")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")

    # --------------------------------------------------------
    # CREATE DATASETS
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset, val_dataset, test_dataset = create_datasets(
        train_transform=train_transform,
        validation_transform=validation_transform
    )

    print(
        f"Train samples      : {len(train_dataset)}"
    )

    print(
        f"Validation samples : {len(val_dataset)}"
    )

    print(
        f"Test samples       : {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # CREATE DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(
        f"Train batches      : {len(train_loader)}"
    )

    print(
        f"Validation batches : {len(val_loader)}"
    )

    print(
        f"Test batches       : {len(test_loader)}"
    )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print("\nCreating EfficientNet-B0 model...")

    model = create_efficientnet_b0(
        num_classes=NUM_CLASSES,
        pretrained=True
    )

    model = model.to(DEVICE)

    print("Model created successfully.")

    print(model.classifier)

    # --------------------------------------------------------
    # LOSS FUNCTION
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # BEST MODEL TRACKING
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_epoch = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(1, EPOCHS + 1):

        # ====================================================
        # TRAINING
        # ====================================================

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        train_loss = (
            running_loss / total
        )

        train_accuracy = (
            correct / total
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_running_loss += (
                    loss.item() * images.size(0)
                )

                _, predicted = torch.max(
                    outputs,
                    1
                )

                val_total += labels.size(0)

                val_correct += (
                    predicted == labels
                ).sum().item()

        val_loss = (
            val_running_loss / val_total
        )

        val_accuracy = (
            val_correct / val_total
        )

        # ====================================================
        # PRINT EPOCH RESULTS
        # ====================================================

        print(
            f"\nEpoch [{epoch}/{EPOCHS}]"
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Acc:    {val_accuracy:.4f}"
        )

        # ====================================================
        # SAVE BEST CHECKPOINT
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                },
                CHECKPOINT_PATH
            )

            print(
                f"Best model saved "
                f"(epoch {epoch})"
            )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation loss : "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best checkpoint epoch: "
        f"{best_epoch}"
    )

    print(
        f"Checkpoint saved at  : "
        f"{CHECKPOINT_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()