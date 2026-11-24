import sys
from pathlib import Path

import torch


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.classification.resnet50 import create_resnet50


print("=" * 60)
print("DERMAI RESNET-50 MODEL TEST")
print("=" * 60)


# ---------------------------------------------------------
# CREATE MODEL
# ---------------------------------------------------------

print("\nCreating ResNet-50...")

model = create_resnet50(
    num_classes=6,
    pretrained=True
)


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

print("\nModel created successfully.")

print(f"Final classification layer:")
print(model.fc)


# ---------------------------------------------------------
# TEST INPUT
# ---------------------------------------------------------

dummy_input = torch.randn(
    2,
    3,
    224,
    224
)

print("\nDummy input shape:")
print(dummy_input.shape)


# ---------------------------------------------------------
# FORWARD PASS
# ---------------------------------------------------------

print("\nRunning forward pass...")

with torch.no_grad():
    output = model(dummy_input)


print("\nOutput shape:")
print(output.shape)


# ---------------------------------------------------------
# EXPECTED OUTPUT
# ---------------------------------------------------------

expected_shape = (2, 6)

if tuple(output.shape) == expected_shape:
    print("\nPASS: Model output shape is correct.")
else:
    print("\nFAIL: Unexpected model output shape.")


print("\n" + "=" * 60)
print("RESNET-50 TEST COMPLETE")
print("=" * 60)