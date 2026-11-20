from __future__ import annotations

from typing import Any

# Clinical mapping of HAM10000 disease categories to severity risk profiles
DISEASE_RISK_PROFILES: dict[str, dict[str, Any]] = {
    "mel": {
        "full_name": "Melanoma",
        "description": "Malignant neoplasm of melanocytes with high metastatic potential.",
        "base_risk": "High Risk",
        "severity_score": 92,
        "urgency": "Urgent",
        "clinical_category": "Malignant Skin Neoplasm",
        "recommended_action": "Urgent dermatologist referral for dermoscopic evaluation and potential full-thickness excisional biopsy.",
        "risk_factors": [
            "High metastatic potential",
            "Malignant melanocytic pathology",
            "Requires histopathological confirmation",
        ],
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "description": "Common non-melanoma skin cancer originating from basal cells in the epidermis.",
        "base_risk": "High Risk",
        "severity_score": 80,
        "urgency": "High",
        "clinical_category": "Malignant Skin Neoplasm (Non-Melanoma)",
        "recommended_action": "Prompt consultation with a dermatologist for margin-controlled excision (e.g. Mohs surgery) or topical/cryotherapy depending on subtype.",
        "risk_factors": [
            "Locally invasive behavior",
            "Potential for tissue destruction if untreated",
            "Surgical excision typically indicated",
        ],
    },
    "akiec": {
        "full_name": "Actinic Keratosis / Bowen's Disease",
        "description": "Precancerous squamous lesion or intraepithelial squamous cell carcinoma in situ induced by UV radiation.",
        "base_risk": "Moderate Risk",
        "severity_score": 62,
        "urgency": "Moderate",
        "clinical_category": "Premalignant / In-Situ Keratinocytic Lesion",
        "recommended_action": "Clinical evaluation for cryotherapy, photodynamic therapy (PDT), or topical fluorouracil/imiquimod to prevent progression.",
        "risk_factors": [
            "Potential progression to invasive squamous cell carcinoma (SCC)",
            "Associated with chronic ultraviolet radiation damage",
            "Requires field treatment or lesion destruction",
        ],
    },
    "bkl": {
        "full_name": "Benign Keratosis",
        "description": "Benign epidermal proliferation including seborrheic keratosis, solar lentigo, and lichen-planus-like keratosis.",
        "base_risk": "Low Risk",
        "severity_score": 25,
        "urgency": "Routine",
        "clinical_category": "Benign Keratinocytic Lesion",
        "recommended_action": "Routine observation. Treatment (e.g., cryotherapy or curettage) is elective for irritation or aesthetic preference.",
        "risk_factors": [
            "Benign nature without metastatic risk",
            "May mimic malignant pigmented lesions clinically",
        ],
    },
    "df": {
        "full_name": "Dermatofibroma",
        "description": "Common benign dermal fibrohistiocytic proliferation characterized by firm dermal nodule with positive 'dimple sign'.",
        "base_risk": "Low Risk",
        "severity_score": 18,
        "urgency": "Routine",
        "clinical_category": "Benign Dermal Fibrohistiocytic Lesion",
        "recommended_action": "Reassurance and observation. Excision is rarely required unless symptomatic or painful.",
        "risk_factors": [
            "Benign cutaneous nodule",
            "Indolent natural history",
        ],
    },
    "nv": {
        "full_name": "Melanocytic Nevus",
        "description": "Benign melanocytic neoplasm (common mole) composed of clustered nevus cells.",
        "base_risk": "Low Risk",
        "severity_score": 20,
        "urgency": "Routine",
        "clinical_category": "Benign Melanocytic Lesion",
        "recommended_action": "Self-monitoring according to ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolution). Periodic skin check recommended.",
        "risk_factors": [
            "Benign melanocytic proliferation",
            "Follow-up recommended if change in size, shape, or color is observed",
        ],
    },
    "vasc": {
        "full_name": "Vascular Lesion",
        "description": "Benign vascular malformation or proliferation (e.g., cherry angioma, pyogenic granuloma, angiokeratoma).",
        "base_risk": "Low Risk",
        "severity_score": 22,
        "urgency": "Routine",
        "clinical_category": "Benign Vascular Anomaly",
        "recommended_action": "Routine observation. Laser ablation or electrodessication may be performed if recurrent bleeding or irritation occurs.",
        "risk_factors": [
            "Benign vascular lesion",
            "Potential for minor hemorrhage if traumatized",
        ],
    },
}

FALLBACK_PROFILE: dict[str, Any] = {
    "full_name": "Unspecified Lesion",
    "description": "Skin lesion requiring formal clinical evaluation.",
    "base_risk": "Requires Clinical Review",
    "severity_score": 50,
    "urgency": "Moderate",
    "clinical_category": "Indeterminate Lesion",
    "recommended_action": "Comprehensive in-person evaluation by a licensed dermatologist.",
    "risk_factors": ["Ambiguous or unverified lesion class"],
}


def assess_skin_disease_risk(
    predicted_class: str | None,
    confidence: float | None,
    quality_score: float | None = None,
    is_poor_quality: bool = False,
) -> dict[str, Any]:
    """
    Computes disease severity and clinical risk classification adhering to
    the DermAI specification (Section 8 & Milestone 3 requirements).
    """
    if not predicted_class:
        return {
            "risk_level": "Requires Clinical Review",
            "severity_score": 50,
            "urgency": "Moderate",
            "clinical_name": "Undetermined Condition",
            "clinical_category": "Indeterminate",
            "recommended_action": "Formal clinical review is recommended due to absence of classification.",
            "risk_factors": ["No valid model prediction was generated."],
            "disclaimer": (
                "DermAI risk categorization is an AI-assisted screening decision support "
                "tool and does NOT constitute an autonomous medical diagnosis or clinical prescription."
            ),
        }

    profile = DISEASE_RISK_PROFILES.get(predicted_class.lower(), FALLBACK_PROFILE)
    conf = float(confidence) if confidence is not None else 0.0
    base_risk = profile["base_risk"]
    severity = profile["severity_score"]
    urgency = profile["urgency"]
    risk_factors = list(profile["risk_factors"])

    # Modifiers based on confidence and image quality:
    final_risk = base_risk

    if is_poor_quality or (quality_score is not None and quality_score < 40.0):
        final_risk = "Requires Clinical Review"
        urgency = "High" if base_risk == "High Risk" else "Moderate"
        risk_factors.append("Image quality assessment indicates low clarity or severe blur.")

    elif conf < 0.50:
        final_risk = "Requires Clinical Review"
        risk_factors.append(f"Model prediction confidence is low ({conf * 100:.1f}%).")

    elif base_risk == "High Risk" and conf >= 0.70:
        final_risk = "High Risk"
        urgency = "Urgent"
        risk_factors.append(f"High-confidence prediction ({conf * 100:.1f}%) for high-risk lesion.")

    return {
        "risk_level": final_risk,
        "base_risk": base_risk,
        "severity_score": severity,
        "urgency": urgency,
        "clinical_name": profile["full_name"],
        "clinical_category": profile["clinical_category"],
        "description": profile["description"],
        "recommended_action": profile["recommended_action"],
        "risk_factors": risk_factors,
        "confidence_level": (
            "High Confidence" if conf >= 0.80 else "Moderate Confidence" if conf >= 0.50 else "Low Confidence"
        ),
        "disclaimer": (
            "DermAI risk categorization is an AI-assisted screening decision support "
            "tool and does NOT constitute an autonomous medical diagnosis or clinical prescription."
        ),
    }

