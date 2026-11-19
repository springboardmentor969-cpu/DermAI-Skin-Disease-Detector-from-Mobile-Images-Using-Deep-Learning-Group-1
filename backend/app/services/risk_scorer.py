"""
Clinical Risk Scorer Service for DermAI Skin Lesion Screening.

Implements a 3-tier risk stratification engine based on predicted dermatological
pathology and model confidence thresholds:
- HIGH: Urgent dermatologist review (urgency: 1 day, color: red)
- MEDIUM: Clinical consultation recommended (urgency: 14 days, color: amber)
- LOW: Routine monitoring, no urgent action (urgency: 90 days, color: green)
"""

from typing import Dict, Any, Literal

RiskLevel = Literal["HIGH", "MEDIUM", "LOW"]
RiskColor = Literal["red", "amber", "green"]
UrgencyDays = Literal[1, 14, 90]


def compute_risk(predicted_class: str, confidence: float) -> Dict[str, Any]:
    """
    Computes 3-tier clinical risk score from predicted diagnostic code and model confidence.

    Args:
        predicted_class: Diagnostic code string (e.g. 'mel', 'bcc', 'akiec', 'bkl', 'nv', 'df', 'vasc').
        confidence: Softmax probability confidence score between 0.0 and 1.0.

    Returns:
        Dictionary containing:
            - risk_level: 'HIGH' | 'MEDIUM' | 'LOW'
            - risk_color: 'red' | 'amber' | 'green'
            - risk_message: Human-readable advice and clinical guidance
            - urgency_days: Recommended follow-up window in days (1, 14, or 90)
    """
    cls = str(predicted_class or "").lower().strip()
    conf = float(confidence)

    # 1. HIGH TIER (urgent dermatologist review: within 1 day)
    # - Melanoma (mel) at any confidence
    if cls == "mel":
        return {
            "risk_level": "HIGH",
            "risk_color": "red",
            "risk_message": "Urgent dermatologist review recommended. High clinical priority for suspected melanoma.",
            "urgency_days": 1,
        }

    # - Basal cell carcinoma (bcc) or actinic keratosis (akiec) at confidence >= 0.5
    if cls in ("bcc", "akiec") and conf >= 0.5:
        lesion_type = "basal cell carcinoma" if cls == "bcc" else "actinic keratosis / Bowen's disease"
        return {
            "risk_level": "HIGH",
            "risk_color": "red",
            "risk_message": f"Urgent dermatologist review recommended. Substantial likelihood of {lesion_type}.",
            "urgency_days": 1,
        }

    # 2. SPECIFIC LESION CLASS EVALUATION
    # - bcc or akiec at confidence < 0.5 -> MEDIUM
    if cls in ("bcc", "akiec") and conf < 0.5:
        return {
            "risk_level": "MEDIUM",
            "risk_color": "amber",
            "risk_message": "Clinical consultation recommended. Potential pre-malignant or carcinoma features with moderate model confidence.",
            "urgency_days": 14,
        }

    # - bkl (benign keratosis): >= 0.4 is LOW, < 0.4 is MEDIUM
    if cls == "bkl":
        if conf >= 0.4:
            return {
                "risk_level": "LOW",
                "risk_color": "green",
                "risk_message": "Routine monitoring recommended. Features consistent with benign keratosis.",
                "urgency_days": 90,
            }
        else:
            return {
                "risk_level": "MEDIUM",
                "risk_color": "amber",
                "risk_message": "Clinical consultation recommended. Low confidence in benign keratosis classification.",
                "urgency_days": 14,
            }

    # - nv, df, vasc (benign nevi, dermatofibroma, vascular): >= 0.5 is LOW, < 0.5 is MEDIUM
    if cls in ("nv", "df", "vasc"):
        if conf >= 0.5:
            name_map = {
                "nv": "melanocytic nevus (mole)",
                "df": "dermatofibroma",
                "vasc": "vascular lesion",
            }
            lesion_desc = name_map.get(cls, "benign lesion")
            return {
                "risk_level": "LOW",
                "risk_color": "green",
                "risk_message": f"Routine monitoring recommended. Features consistent with a {lesion_desc}. No urgent action needed.",
                "urgency_days": 90,
            }
        else:
            return {
                "risk_level": "MEDIUM",
                "risk_color": "amber",
                "risk_message": "Clinical consultation recommended. Low model certainty requires professional examination.",
                "urgency_days": 14,
            }

    # 3. GENERIC FALLBACK FOR UNLISTED CLASSES
    # - Any class where confidence < 0.45 (model uncertain) -> MEDIUM
    if conf < 0.45:
        return {
            "risk_level": "MEDIUM",
            "risk_color": "amber",
            "risk_message": "Clinical consultation recommended. Low model certainty warrants in-person professional evaluation.",
            "urgency_days": 14,
        }

    return {
        "risk_level": "LOW",
        "risk_color": "green",
        "risk_message": "Routine monitoring recommended. No urgent clinical indicators detected.",
        "urgency_days": 90,
    }

