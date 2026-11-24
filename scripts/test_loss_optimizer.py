import sys
from pathlib import Path


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch

from models.classification.resnet50 import create_resnet50
from training.loss_optimizer import (
    DEVICE,
    create_loss_function,
    create_optimizer
)


print("=" * 60)
print("DERMAI LOSS AND OPTIMIZER TEST")
print("=" * 60)


# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

print("\nDevice:")
print(DEVICE)


# ---------------------------------------------------------
# CREATE MODEL
# ---------------------------------------------------------

print("\nCreating ResNet-50...")

model = create_resnet50(
    num_classes=6,
    pretrained=True
)

model = model.to(DEVICE)

print("Model created successfully.")


# ---------------------------------------------------------
# CREATE LOSS
# ---------------------------------------------------------

print("\nCreating class-weighted loss...")

criterion = create_loss_function()

print(criterion)


# ---------------------------------------------------------
# CREATE OPTIMIZER
# ---------------------------------------------------------

print("\nCreating AdamW optimizer...")

optimizer = create_optimizer(model)

print(optimizer)


# ---------------------------------------------------------
# TEST LOSS
# ---------------------------------------------------------

dummy_output = torch.randn(
    4,
    6,
    device=DEVICE
)

dummy_labels = torch.tensor(
    [0, 1, 2, 3],
    device=DEVICE
)

loss = criterion(
    dummy_output,
    dummy_labels
)

print("\nDummy loss:")
print(f"{loss.item():.4f}")


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("LOSS AND OPTIMIZER TEST COMPLETE")
print("=" * 60)