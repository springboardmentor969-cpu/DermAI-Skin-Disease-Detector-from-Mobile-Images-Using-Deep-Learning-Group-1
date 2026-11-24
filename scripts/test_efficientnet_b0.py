import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.classification.efficientnet_b0 import create_efficientnet_b0


model = create_efficientnet_b0(
    num_classes=6,
    pretrained=True
)

print("EfficientNet-B0 created successfully.")
print(model.classifier)