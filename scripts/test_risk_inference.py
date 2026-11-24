
from pathlib import Path
import sys

DERMAI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DERMAI_ROOT))

from PIL import Image
from backend.services.inference import predict_image

IMAGE_PATH = Path(
    r"C:\Users\SRINJOY\Dropbox\PC\Downloads\PAD-UFES-20\images\imgs_part_1\PAT_9_17_80.png"
)

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image = Image.open(IMAGE_PATH).convert("RGB")

result = predict_image(image)

print("\n" + "=" * 60)
print("DERMAI RISK-INTEGRATED PREDICTION TEST")
print("=" * 60)

print("Condition       :", result["condition"])
print("Confidence      :", result["confidence"])
print("Risk category   :", result["risk_category"])
print("Review priority :", result["review_priority"])
print("Assessment type :", result["assessment_type"])
print("Risk message    :", result["risk_message"])
print("Disclaimer      :", result["medical_disclaimer"])

print("=" * 60)

required_fields = [
    "risk_assessment",
    "risk_category",
    "review_priority",
    "assessment_type",
    "risk_message",
    "medical_disclaimer"
]

assert all(field in result for field in required_fields)

print("SUCCESS: Risk assessment integrated with inference.")