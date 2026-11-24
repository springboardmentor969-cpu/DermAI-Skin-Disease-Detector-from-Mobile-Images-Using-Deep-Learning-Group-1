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
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader

from models.classification.resnet50 import create_resnet50
from dataset.dataset_loader import create_datasets
from training.data_transforms import validation_transform
from training.training_config import BEST_MODEL_PATH


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK"
]

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = OUTPUT_DIR / "resnet50_confusion_matrix.png"


# --------------------------------------------------
# LOAD TEST DATASET
# --------------------------------------------------

print("=" * 60)
print("DERMAI RESNET-50 CONFUSION MATRIX")
print("=" * 60)

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


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

print("\nCreating ResNet-50 model...")

model = create_resnet50(
    num_classes=6,
    pretrained=False
)

model = model.to(DEVICE)


# --------------------------------------------------
# LOAD CHECKPOINT
# --------------------------------------------------

print("\nLoading best checkpoint...")

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

model.eval()


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion matrix:")
print(cm)


# --------------------------------------------------
# PLOT
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 7)
)

image = ax.imshow(
    cm,
    interpolation="nearest"
)

fig.colorbar(image, ax=ax)

ax.set(
    xticks=range(len(CLASS_NAMES)),
    yticks=range(len(CLASS_NAMES)),
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    xlabel="Predicted Label",
    ylabel="Actual Label",
    title="ResNet-50 Confusion Matrix - PAD-UFES-20"
)

# Add numbers inside cells
threshold = cm.max() / 2

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )


fig.tight_layout()


# --------------------------------------------------
# SAVE FIGURE
# --------------------------------------------------

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\n" + "=" * 60)
print("CONFUSION MATRIX SAVED")
print("=" * 60)

print(f"\nSaved to:")
print(OUTPUT_PATH)

print("\nCompleted successfully.")