import torch

from models.classification.vit import create_vit


print("=" * 60)
print("DERMAI VISION TRANSFORMER MODEL TEST")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

model = create_vit(
    num_classes=6,
    pretrained=True
)

model = model.to(device)

print("\nModel created successfully.")
print("Model:", model.__class__.__name__)

print("\nClassification head:")
print(model.heads.head)

dummy_input = torch.randn(
    1,
    3,
    224,
    224
).to(device)

with torch.no_grad():
    output = model(dummy_input)

print("\nInput shape :", dummy_input.shape)
print("Output shape:", output.shape)

print("\nExpected output shape: [1, 6]")

if output.shape == (1, 6):
    print("\nSUCCESS: ViT model is working correctly.")
else:
    print("\nERROR: Unexpected output shape.")

print("=" * 60)