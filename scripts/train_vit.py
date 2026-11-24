import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataset_loader import create_datasets
from models.classification.vit import create_vit


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

NUM_CLASSES = 6

CHECKPOINT_PATH = (
    "models/checkpoints/vit_b16_best.pth"
)

CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK"
]

# Same class weights used for the CNN models
CLASS_WEIGHTS = torch.tensor(
    [
        0.5177,  # ACK
        0.4592,  # BCC
        7.5648,  # MEL
        1.5473,  # NEV
        2.0323,  # SCC
        1.6114   # SEK
    ],
    dtype=torch.float32
).to(DEVICE)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):
    model.train()

    running_loss = 0.0
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

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    loader,
    criterion
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DERMAI VISION TRANSFORMER (ViT-B/16) TRAINING")
    print("=" * 70)

    print("\nDevice:")
    print(DEVICE)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset, val_dataset, test_dataset = create_datasets()

    print("Train samples :", len(train_dataset))
    print("Val samples   :", len(val_dataset))
    print("Test samples  :", len(test_dataset))

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

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

    print("\nTrain batches:", len(train_loader))
    print("Val batches  :", len(val_loader))

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nCreating ViT-B/16 model...")

    model = create_vit(
        num_classes=NUM_CLASSES,
        pretrained=True
    )

    model = model.to(DEVICE)

    print("Model created successfully.")

    print("\nClassification head:")
    print(model.heads.head)

    # --------------------------------------------------------
    # LOSS
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
    # TRAINING
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_epoch = 0

    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

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

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "class_names": CLASS_NAMES
                },
                CHECKPOINT_PATH
            )

            print(
                f"  -> Best model saved "
                f"(epoch {epoch})"
            )

    print("-" * 70)

    print("\nTraining completed.")

    print("Best epoch     :", best_epoch)
    print("Best val loss  :", f"{best_val_loss:.4f}")
    print("Checkpoint     :", CHECKPOINT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()