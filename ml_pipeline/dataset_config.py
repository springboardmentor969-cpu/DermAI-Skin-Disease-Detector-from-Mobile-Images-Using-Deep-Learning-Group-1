'''
DermAI Milestone 1 — Target Skin Conditions & Public/De-Identified Dataset Specifications.
Covers HAM10000 (Dermoscopic) and PAD-UFES-20 (Smartphone Clinical Images).
'''

DATASETS = {
    "ham10000": {
        "name": "HAM10000 (Human Against Machine with 10000 training images)",
        "modality": "Dermoscopic & Clinical Close-Up",
        "total_images": 10015,
        "annotation_type": "Histopathology (>50%), Confocal Microscopy, Expert Consensus",
        "source_url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T",
        "classes": [
            {"code": "nv", "name": "Melanocytic Nevus", "count": 6705, "risk_tier": "Low Risk"},
            {"code": "mel", "name": "Melanoma", "count": 1113, "risk_tier": "High Risk (Requires Clinical Review)"},
            {"code": "bkl", "name": "Benign Keratosis-like Lesions", "count": 1099, "risk_tier": "Low Risk"},
            {"code": "bcc", "name": "Basal Cell Carcinoma", "count": 514, "risk_tier": "High Risk (Requires Clinical Review)"},
            {"code": "akiec", "name": "Actinic Keratoses / Bowen's Disease", "count": 327, "risk_tier": "Moderate-High Risk"},
            {"code": "vasc", "name": "Vascular Lesions", "count": 142, "risk_tier": "Low-Moderate Risk"},
            {"code": "df", "name": "Dermatofibroma", "count": 115, "risk_tier": "Low Risk"},
        ],
        "anatomical_sites": {
            "Back": 2192,
            "Lower Extremity": 2077,
            "Trunk": 1404,
            "Upper Extremity": 1118,
            "Abdomen": 1022,
            "Face": 745,
            "Chest": 407,
            "Foot / Acral / Other": 1050,
        },
    },
    "pad_ufes_20": {
        "name": "PAD-UFES-20 (Smartphone-Captured Skin Lesion Dataset)",
        "modality": "Mobile Smartphone Camera Images + Patient Clinical Metadata",
        "total_images": 2298,
        "annotation_type": "Biopsy-proven Malignancies + Clinical Consensus",
        "source_url": "https://data.mendeley.com/datasets/zr7vgbcyr2/1",
        "classes": [
            {"code": "BCC", "name": "Basal Cell Carcinoma", "count": 845, "risk_tier": "High Risk (Requires Clinical Review)"},
            {"code": "ACK", "name": "Actinic Keratosis", "count": 730, "risk_tier": "Moderate-High Risk"},
            {"code": "NEV", "name": "Melanocytic Nevus", "count": 244, "risk_tier": "Low Risk"},
            {"code": "SEK", "name": "Seborrheic Keratosis", "count": 235, "risk_tier": "Low Risk"},
            {"code": "SCC", "name": "Squamous Cell Carcinoma", "count": 192, "risk_tier": "High Risk (Requires Clinical Review)"},
            {"code": "MEL", "name": "Melanoma", "count": 52, "risk_tier": "High Risk (Requires Clinical Review)"},
        ],
        "anatomical_sites": {
            "Face / Head / Neck": 1124,
            "Forearm / Hand": 518,
            "Chest / Trunk": 342,
            "Lower Limb": 198,
            "Back / Shoulder": 116,
        },
    },
}
