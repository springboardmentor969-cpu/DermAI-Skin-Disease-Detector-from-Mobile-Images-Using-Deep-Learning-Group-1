
from pathlib import Path
import sys

DERMAI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DERMAI_ROOT))

from backend.services.risk_assessment import assess_risk


test_cases = [
    ("MEL", 0.92),
    ("SCC", 0.85),
    ("BCC", 0.88),
    ("ACK", 0.78),
    ("NEV", 0.90),
    ("SEK", 0.91),
    ("MEL", 0.42),
    ("UNKNOWN", 0.90),
]

print("=" * 65)
print("DERMAI RISK ASSESSMENT TEST")
print("=" * 65)

for condition, confidence in test_cases:
    result = assess_risk(condition, confidence)

    print(f"\nCondition       : {result['condition']}")
    print(f"Confidence      : {result['confidence_percentage']}%")
    print(f"Risk category   : {result['risk_category']}")
    print(f"Review priority : {result['review_priority']}")
    print(f"Message         : {result['message']}")

print("\n" + "=" * 65)
print("RISK ASSESSMENT TEST COMPLETED")
print("=" * 65)