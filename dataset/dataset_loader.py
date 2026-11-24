from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from preprocessing.augmentation import (
    get_train_transforms,
    get_validation_transforms,
    get_test_transforms
)


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS_DIR = PROJECT_ROOT / "splits"
PROCESSED_IMAGES_DIR = PROJECT_ROOT / "processed_images"


# ---------------------------------------------------------
# CLASS DEFINITIONS
# ---------------------------------------------------------

CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK"
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


# ---------------------------------------------------------
# PAD-UFES-20 PYTORCH DATASET
# ---------------------------------------------------------

class PADUFES20Dataset(Dataset):

    def __init__(self, csv_path, transform=None):

        self.csv_path = Path(csv_path)
        self.transform = transform

        # Load split CSV
        self.data = pd.read_csv(self.csv_path)

        # Validate diagnostic classes
        unknown_classes = set(self.data["diagnostic"]) - set(CLASS_NAMES)

        if unknown_classes:
            raise ValueError(
                f"Unknown diagnostic classes found: {unknown_classes}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        # Image path
        image_path = Path(
            str(row["image_path"]).replace(
                r"C:\Users\SRINJOY\OneDrive\Desktop\DermAI",
                str(PROJECT_ROOT)
            )
        )

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Disease label
        diagnostic = row["diagnostic"]

        label = CLASS_TO_INDEX[diagnostic]

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------
# CREATE TRAIN / VALIDATION / TEST DATASETS
# ---------------------------------------------------------

def create_datasets(
    train_transform=None,
    validation_transform=None,
    test_transform=None
):

    # Use augmentation/transforms automatically if not supplied
    if train_transform is None:
        train_transform = get_train_transforms()

    if validation_transform is None:
        validation_transform = get_validation_transforms()

    if test_transform is None:
        test_transform = get_test_transforms()

    # Split CSV files
    train_csv = SPLITS_DIR / "train.csv"
    validation_csv = SPLITS_DIR / "validation.csv"
    test_csv = SPLITS_DIR / "test.csv"

    # Create datasets
    train_dataset = PADUFES20Dataset(
        train_csv,
        transform=train_transform
    )

    validation_dataset = PADUFES20Dataset(
        validation_csv,
        transform=validation_transform
    )

    test_dataset = PADUFES20Dataset(
        test_csv,
        transform=test_transform
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )