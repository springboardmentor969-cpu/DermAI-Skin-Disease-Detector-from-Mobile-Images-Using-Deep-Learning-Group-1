import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from torch.utils.data import DataLoader

from models.classification.densenet121 import create_densenet121
from dataset.dataset_loader import create_datasets
from training.data_transforms import train_transform, validation_transform
from training.class_weights import calculate_class_weights
from training.training_config import (
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CHECKPOINT_DIR
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DENSENET_CHECKPOINT_PATH = CHECKPOINT_DIR / "densenet121_best.pth"


print("=" * 60)
print("DERMAI DENSENET-121 TRAINING")
print("=" * 60)

print("\nDevice:")
print(DEVICE)


# ------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------

print("\nLoading datasets...")

train_dataset, validation_dataset, _ = create_datasets(
    train_transform=train_transform,
    validation_transform=validation_transform
)

print(f"Train samples      : {len(train_dataset)}")
print(f"Validation samples : {len(validation_dataset)}")


# ------------------------------------------------------------
# DATALOADERS
# ------------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print(f"Train batches      : {len(train_loader)}")
print(f"Validation batches : {len(validation_loader)}")


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

print("\nCreating DenseNet-121 model...")

model = create_densenet121(
    num_classes=6,
    pretrained=True
)

model = model.to(DEVICE)

print("DenseNet-121 created successfully.")


# ------------------------------------------------------------
# CLASS-WEIGHTED LOSS
# ------------------------------------------------------------

print("\nCalculating class weights...")

_, class_weights = calculate_class_weights()

class_weights = class_weights.to(DEVICE)

print("Class weights:")
print(class_weights)

criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights
)


# ------------------------------------------------------------
# OPTIMIZER
# ------------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ------------------------------------------------------------
# CHECKPOINT DIRECTORY
# ------------------------------------------------------------

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

best_validation_loss = float("inf")


for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("=" * 60)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

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

        running_train_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct_train += (
            (predictions == labels)
            .sum()
            .item()
        )

        total_train += labels.size(0)

    train_loss = (
        running_train_loss /
        total_train
    )

    train_accuracy = (
        correct_train /
        total_train
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    running_validation_loss = 0.0
    correct_validation = 0
    total_validation = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_validation_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct_validation += (
                (predictions == labels)
                .sum()
                .item()
            )

            total_validation += labels.size(0)

    validation_loss = (
        running_validation_loss /
        total_validation
    )

    validation_accuracy = (
        correct_validation /
        total_validation
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: {validation_loss:.4f} | "
        f"Validation Accuracy: {validation_accuracy:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_loss": train_loss,
                "training_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy
            },
            DENSENET_CHECKPOINT_PATH
        )

        print("\nBest DenseNet-121 model saved.")


# ------------------------------------------------------------
# COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DENSENET-121 TRAINING COMPLETED")
print("=" * 60)

print("\nBest validation loss:")
print(f"{best_validation_loss:.4f}")

print("\nBest model saved to:")
print(DENSENET_CHECKPOINT_PATH)