import sys
from pathlib import Path

import torch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_loader import create_datasets


print("=" * 60)
print("DERMAI DATASET LOADER TEST")
print("=" * 60)

print("\nCreating datasets...")

train_dataset, validation_dataset, test_dataset = create_datasets()

print(f"Training samples   : {len(train_dataset)}")
print(f"Validation samples : {len(validation_dataset)}")
print(f"Test samples       : {len(test_dataset)}")


print("\n" + "=" * 60)
print("TESTING TRAINING SAMPLE")
print("=" * 60)

image, label = train_dataset[0]

print(f"Image type  : {type(image)}")
print(f"Label type  : {type(label)}")
print(f"Label value : {label}")

if hasattr(image, "size"):
    print(f"Image size  : {image.size}")

print("\n" + "=" * 60)
print("DATASET LOADER TEST COMPLETE")
print("=" * 60)