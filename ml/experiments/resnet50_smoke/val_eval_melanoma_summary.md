# Melanoma & Confusion Analysis — resnet50 (val split)

## Key Melanoma Metrics

| Metric | Value |
|---|---|
| Melanoma Sensitivity (Recall) | **0.2235** |
| Melanoma Specificity | 0.7841 |
| Melanoma Precision | 0.1145 |
| Melanoma AUC-ROC (OvR) | 0.5224 |
| Overall Macro-F1 | 0.1355 |
| Overall Balanced Accuracy | 0.1494 |

> **Why melanoma recall matters:** A false negative (missed melanoma) is clinically
> far more dangerous than a false positive. We want sensitivity as close to 1.0 as
> possible, even if precision suffers.

## mel ↔ nv Confusion

| | Count | % of true class |
|---|---|---|
| mel correctly predicted as **mel** | 38 | 22.4% |
| mel **missed** and predicted as nv | **101** | **59.4%** |
| mel predicted as other classes | 31 | 18.2% |
| nv falsely predicted as mel | 219 | 21.2% |

## Interpretation

- ❌ **Melanoma recall is LOW** (< 0.60). The model misses most melanoma cases — this is the #1 priority to fix.
- ❌ **mel→nv confusion is high** (59.4% of melanoma predicted as benign nevi). Melanoma and common nevi share dermoscopic features (pigmentation patterns, borders). This is the most dangerous error type — a melanoma dismissed as a common mole.
- ⚠️ **nv→mel false alarms** (21.2% of nevi predicted as melanoma). High false positive rate means unnecessary referrals but is clinically safer than missed mel.

## All Per-Class Sensitivity / Specificity

| Class | Sensitivity | Specificity | AUC-ROC | Support |
|---|---|---|---|---|
| **akiec** | 0.0000 | 0.9986 | 0.6555 | 51 |
| **bcc** | 0.0000 | 0.9897 | 0.5906 | 77 |
| **bkl** | 0.0892 | 0.9425 | 0.5859 | 157 |
| **df** | 0.1053 | 0.8890 | 0.4348 | 19 |
| **mel** ⚠️ | 0.2235 | 0.7841 | 0.5224 | 170 |
| **nv** | 0.6277 | 0.4558 | 0.6104 | 1034 |
| **vasc** | 0.0000 | 1.0000 | 0.5539 | 24 |
