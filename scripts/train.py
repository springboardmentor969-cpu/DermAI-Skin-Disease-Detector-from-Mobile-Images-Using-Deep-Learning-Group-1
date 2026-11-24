
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
)

# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------

from models.classification.resnet50 import create_resnet50

from training.data_loader import (
    train_loader,
    validation_loader
)

from training.loss_optimizer import (
    DEVICE,
    create_loss_function,
    create_optimizer
)

from training.training_config import (
    NUM_CLASSES,
    NUM_EPOCHS,
)


# ---------------------------------------------------------
# CHECKPOINT PATH
# ---------------------------------------------------------

TUNED_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "resnet50_tuned_best.pth"
)

TUNED_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

print("=" * 60)
print("DERMAI RESNET-50 FINAL TUNED TRAINING")
print("=" * 60)

print("\nDevice:")
print(DEVICE)

print("\nSelected hyperparameters:")
print("Optimizer       : AdamW")
print("Learning rate   : 0.0001")
print("Weight decay    : 0.0001")
print("Batch size      : 32")
print("Epochs          :", NUM_EPOCHS)

print("\nCreating pretrained ResNet-50...")

model = create_resnet50(
    num_classes=NUM_CLASSES,
    pretrained=True
)

model = model.to(DEVICE)

print("Model created successfully.")


# ---------------------------------------------------------
# LOSS AND OPTIMIZER
# ---------------------------------------------------------

criterion = create_loss_function()

optimizer = create_optimizer(model)


# ---------------------------------------------------------
# TRAINING VARIABLES
# ---------------------------------------------------------

best_validation_loss = float("inf")
best_validation_accuracy = 0.0
best_macro_f1 = 0.0
best_balanced_accuracy = 0.0
best_epoch = 0


# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)
    print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
    print("=" * 60)

    # -----------------------------------------------------
    # TRAINING
    # -----------------------------------------------------

    model.train()

    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_train_loss += (
            loss.item() * images.size(0)
        )

        predicted = outputs.argmax(dim=1)

        total_train += labels.size(0)

        correct_train += (
            predicted == labels
        ).sum().item()

        if (batch_index + 1) % 10 == 0:
            print(
                f"Batch {batch_index + 1}/"
                f"{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    # -----------------------------------------------------
    # TRAINING METRICS
    # -----------------------------------------------------

    train_loss = running_train_loss / total_train
    train_accuracy = correct_train / total_train

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    model.eval()

    running_validation_loss = 0.0
    correct_validation = 0
    total_validation = 0

    all_validation_labels = []
    all_validation_predictions = []

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_validation_loss += (
                loss.item() * images.size(0)
            )

            predicted = outputs.argmax(dim=1)

            total_validation += labels.size(0)

            correct_validation += (
                predicted == labels
            ).sum().item()

            all_validation_labels.extend(
                labels.cpu().numpy()
            )

            all_validation_predictions.extend(
                predicted.cpu().numpy()
            )

    # -----------------------------------------------------
    # VALIDATION METRICS
    # -----------------------------------------------------

    validation_loss = (
        running_validation_loss / total_validation
    )

    validation_accuracy = (
        correct_validation / total_validation
    )

    validation_macro_f1 = f1_score(
        all_validation_labels,
        all_validation_predictions,
        average="macro",
        zero_division=0
    )

    validation_balanced_accuracy = balanced_accuracy_score(
        all_validation_labels,
        all_validation_predictions
    )

    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print("\nEpoch results:")

    print(f"Training Loss              : {train_loss:.4f}")
    print(f"Training Accuracy          : {train_accuracy:.4f}")

    print(f"Validation Loss            : {validation_loss:.4f}")
    print(f"Validation Accuracy        : {validation_accuracy:.4f}")

    print(f"Validation Macro F1        : {validation_macro_f1:.4f}")
    print(
        f"Validation Balanced Acc.   : "
        f"{validation_balanced_accuracy:.4f}"
    )

    # -----------------------------------------------------
    # SAVE BEST MODEL
    # Selection criterion: lowest validation loss
    # -----------------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss
        best_validation_accuracy = validation_accuracy
        best_macro_f1 = validation_macro_f1
        best_balanced_accuracy = validation_balanced_accuracy
        best_epoch = epoch + 1

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "learning_rate": 0.0001,
                "weight_decay": 0.0001,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": validation_macro_f1,
                "validation_balanced_accuracy": validation_balanced_accuracy,
            },
            TUNED_MODEL_PATH
        )

        print("\nBest tuned model saved:")
        print(TUNED_MODEL_PATH)


# ---------------------------------------------------------
# TRAINING COMPLETE
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TUNED TRAINING COMPLETE")
print("=" * 60)

print("Best epoch                  :", best_epoch)
print(f"Best validation loss        : {best_validation_loss:.4f}")
print(f"Validation accuracy         : {best_validation_accuracy:.4f}")
print(f"Validation macro F1         : {best_macro_f1:.4f}")
print(f"Validation balanced accuracy: {best_balanced_accuracy:.4f}")

print("\nTuned checkpoint:")
print(TUNED_MODEL_PATH)

print("\nOriginal baseline checkpoint was not overwritten.")
