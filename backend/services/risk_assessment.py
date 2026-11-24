
"""
DermAI Risk Assessment Module

Provides AI-assisted clinical review priority.
This is NOT a medical diagnosis or validated clinical risk score.
"""

REVIEW_RULES = {
    "MEL": {
        "priority": "urgent",
        "category": "high",
        "condition": "Melanoma",
        "message": "Prompt professional dermatological assessment is recommended."
    },
    "SCC": {
        "priority": "urgent",
        "category": "high",
        "condition": "Squamous Cell Carcinoma",
        "message": "Prompt professional dermatological assessment is recommended."
    },
    "BCC": {
        "priority": "urgent",
        "category": "high",
        "condition": "Basal Cell Carcinoma",
        "message": "Professional dermatological assessment is recommended."
    },
    "ACK": {
        "priority": "prompt",
        "category": "moderate",
        "condition": "Actinic Keratosis",
        "message": "Arrange professional dermatological review."
    },
    "NEV": {
        "priority": "routine",
        "category": "low",
        "condition": "Melanocytic Nevus",
        "message": "Routine professional assessment is appropriate."
    },
    "SEK": {
        "priority": "routine",
        "category": "low",
        "condition": "Seborrheic Keratosis",
        "message": "Routine professional assessment is appropriate."
    }
}

CONFIDENCE_THRESHOLD = 0.60


def assess_risk(condition, confidence):
    """
    Determine AI-assisted clinical review priority.

    Args:
        condition: Predicted PAD-UFES-20 class.
        confidence: Model confidence (0-1 or 0-100).

    Returns:
        Dictionary containing review priority and explanation.
    """

    condition = str(condition).strip().upper()

    confidence = float(confidence)

    if confidence > 1:
        confidence /= 100.0

    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1 or 0 and 100.")

    base_result = {
        "predicted_class": condition,
        "confidence": round(confidence, 4),
        "confidence_percentage": round(confidence * 100, 2),
        "assessment_type": "AI-assisted review priority",
        "disclaimer": (
            "This result is not a medical diagnosis or a validated "
            "clinical risk score. Consult a qualified dermatologist."
        )
    }

    # Low confidence: request manual review.
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            **base_result,
            "risk_category": "uncertain",
            "review_priority": "manual_review",
            "condition": REVIEW_RULES.get(
                condition, {}
            ).get("condition", "Uncertain classification"),
            "message": (
                "AI confidence is below 60%. Manual professional "
                "assessment is required. Do not rely on this prediction."
            )
        }

    # Unknown class: never assign a risk category.
    if condition not in REVIEW_RULES:
        return {
            **base_result,
            "risk_category": "uncertain",
            "review_priority": "manual_review",
            "condition": "Unknown",
            "message": "Unrecognized class. Professional review required."
        }

    rule = REVIEW_RULES[condition]

    return {
        **base_result,
        "risk_category": rule["category"],
        "review_priority": rule["priority"],
        "condition": rule["condition"],
        "message": rule["message"]
    }
