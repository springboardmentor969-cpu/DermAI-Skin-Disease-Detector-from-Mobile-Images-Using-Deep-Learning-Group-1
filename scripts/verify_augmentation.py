from pathlib import Path

import matplotlib.pyplot as plt

from dataset.dataset_loader import PADUFES20Dataset
from preprocessing.augmentation import get_train_transforms


# ---------------------------------------------------------
# PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = PROJECT_ROOT / "splits" / "train.csv"


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

dataset = PADUFES20Dataset(
    TRAIN_CSV,
    transform=get_train_transforms()
)


# ---------------------------------------------------------
# GENERATE AUGMENTED VERSIONS
# ---------------------------------------------------------

image1, label1 = dataset[0]
image2, label2 = dataset[0]
image3, label3 = dataset[0]


# ---------------------------------------------------------
# DENORMALIZE FOR DISPLAY
# ---------------------------------------------------------

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def denormalize(image):

    image = image.clone()

    for channel in range(3):
        image[channel] = (
            image[channel] * std[channel]
            + mean[channel]
        )

    return image.clamp(0, 1)


image1 = denormalize(image1)
image2 = denormalize(image2)
image3 = denormalize(image3)


# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image1.permute(1, 2, 0))
plt.title("Augmented Image 1")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(image2.permute(1, 2, 0))
plt.title("Augmented Image 2")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image3.permute(1, 2, 0))
plt.title("Augmented Image 3")
plt.axis("off")

plt.tight_layout()

output_path = PROJECT_ROOT / "outputs" / "augmentation_verification.png"

plt.savefig(output_path, dpi=200)

print("=" * 60)
print("DERMAI AUGMENTATION VERIFICATION")
print("=" * 60)
print(f"Label: {label1}")
print(f"Saved visualization: {output_path}")
print("=" * 60)

plt.show()