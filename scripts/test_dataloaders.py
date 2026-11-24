import sys
from pathlib import Path

import torch


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from training.data_loader import (
    train_loader,
    validation_loader,
    test_loader
)


print("=" * 60)
print("DERMAI DATALOADER TEST")
print("=" * 60)


# ---------------------------------------------------------
# DATASET SIZES
# ---------------------------------------------------------

print("\nDataset sizes:")

print(f"Training   : {len(train_loader.dataset)}")
print(f"Validation : {len(validation_loader.dataset)}")
print(f"Test       : {len(test_loader.dataset)}")


# ---------------------------------------------------------
# NUMBER OF BATCHES
# ---------------------------------------------------------

print("\nNumber of batches:")

print(f"Training   : {len(train_loader)}")
print(f"Validation : {len(validation_loader)}")
print(f"Test       : {len(test_loader)}")


# ---------------------------------------------------------
# GET ONE TRAINING BATCH
# ---------------------------------------------------------

images, labels = next(iter(train_loader))


print("\n" + "=" * 60)
print("TRAINING BATCH")
print("=" * 60)

print(f"Images type : {type(images)}")
print(f"Labels type : {type(labels)}")

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

print(f"Images dtype: {images.dtype}")
print(f"Labels dtype: {labels.dtype}")


# ---------------------------------------------------------
# CHECK LABELS
# ---------------------------------------------------------

print("\nLabels in first batch:")

print(labels)


# ---------------------------------------------------------
# CHECK VALUE RANGE
# ---------------------------------------------------------

print("\nImage tensor statistics:")

print(f"Minimum : {images.min().item():.4f}")
print(f"Maximum : {images.max().item():.4f}")
print(f"Mean    : {images.mean().item():.4f}")
print(f"Std     : {images.std().item():.4f}")


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATALOADER TEST COMPLETE")
print("=" * 60)