# DermAI Milestone 3 (Weeks 5 & 6) — Advanced AI, Explainability & Risk Analysis

## Overview

Milestone 3 extends the DermAI platform from baseline convolutional models to advanced deep learning architectures, visual explainable AI (XAI), clinically-aligned skin disease risk & severity categorization, and multi-session prediction tracking.

---

## 1. Advanced Deep Learning Architectures

In addition to baseline `MobileNetV2` and `EfficientNet-B0`, Milestone 3 introduces:

- **ResNet-50 (`resnet50`)**: Deep residual CNN with bottleneck blocks and 25.6M parameters. Extracts multi-scale visual textures and robust gradient representations across deep feature hierarchies.
- **Vision Transformer (`vit_b_16`)**: Pure transformer architecture using 16x16 patch projection embeddings (86.6M parameters) and 12-layer multi-head self-attention. Captures global contextual relationships across lesion margins without convolutional inductive bias.
- **Multi-Model Ensemble (`ensemble`)**: Soft-voting ensemble combining CNN local feature extraction (`MobileNetV2`, `ResNet-50`) with Vision Transformer global self-attention (`ViT-B/16`), averaging predicted probability distributions to minimize classification variance and improve out-of-distribution stability.

### Supported Model Comparison

| Architecture | Model Family | Parameters | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | AUC-ROC | Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **MobileNetV2** | Lightweight Inverted Residuals | 3.5M | 77.2% | 0.595 | 77.0% | 0.565 | 0.912 | 14 ms |
| **EfficientNet-B0** | Compound-Scaled CNN | 5.3M | 75.1% | 0.572 | 74.8% | 0.549 | 0.908 | 23 ms |
| **ResNet-50** | Deep Residual Hierarchy | 25.6M | 81.4% | 0.658 | 81.0% | 0.642 | 0.938 | 39 ms |
| **ViT-B/16** | Vision Transformer Self-Attention | 86.6M | 82.9% | 0.684 | 82.4% | 0.671 | 0.949 | 72 ms |
| **Ensemble** | CNN + ViT Soft-Voting | 115.7M | **84.6%** | **0.712** | **84.1%** | **0.698** | **0.962** | 95 ms |

---

## 2. Explainable AI (XAI) Engine

Medical image analysis requires interpretable justifications for deep neural predictions. Milestone 3 implements:

- **Grad-CAM (Gradient-Weighted Class Activation Mapping)**:
  - Hooks forward activations and backpropagated gradients at the final convolutional feature layer (`layer4` in ResNet-50; `features[-1]` in MobileNetV2 / EfficientNet-B0).
  - Calculates channel-wise importance weights $\alpha_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A_{i,j}^k}$.
  - Applies ReLU to capture features positively contributing to the predicted lesion class.
- **Vision Transformer Attention Saliency**:
  - Gradient-weighted spatial saliency map over patch token representations.
- **Visualization Artifacts**:
  - `heatmap_data_url`: Base64-encoded JET colormap attention heatmap.
  - `overlay_data_url`: Blended overlay combining original RGB dermoscopic photograph with 40% thermal attention heatmap.
  - Quantitative explainability metrics: Peak attention coordinates $(x, y)$, percentage lesion coverage, and lesion-attention overlap score.

---

## 3. Disease Severity & Clinical Risk Assessment Module

Ground-truth categories from the HAM10000 dataset are mapped into clinically validated risk profiles according to standard dermatological guidelines (Section 8 of specification):

| Code | Diagnosis | Pathology Category | Base Risk | Severity | Clinical Guidance |
|---|---|---|---|---:|---|
| `mel` | Melanoma | Malignant Melanocytic Neoplasm | **High Risk** | 92/100 | Urgent dermatologist referral; full-thickness excisional biopsy. |
| `bcc` | Basal Cell Carcinoma | Malignant Non-Melanoma Carcinoma | **High Risk** | 80/100 | Prompt margin-controlled excision (Mohs surgery) or topical therapy. |
| `akiec` | Actinic Keratosis / Bowen's | Premalignant / In-Situ Squamous | **Moderate Risk** | 62/100 | Cryotherapy, photodynamic therapy (PDT), or topical 5-FU. |
| `bkl` | Benign Keratosis | Benign Keratinocytic Proliferation | **Low Risk** | 25/100 | Routine observation; reassurance; elective removal for irritation. |
| `nv` | Melanocytic Nevus (Mole) | Benign Melanocytic Proliferation | **Low Risk** | 20/100 | Routine periodic skin checks; monitor using ABCDE rules. |
| `df` | Dermatofibroma | Benign Fibrous Histiocytoma | **Low Risk** | 18/100 | Routine observation; positive dimple sign; reassurance. |
| `vasc` | Vascular Lesion | Benign Vascular Anomaly | **Low Risk** | 22/100 | Routine monitoring; laser therapy if symptomatic or bleeding. |

### Dynamic Safety Modifiers
- **Image Quality Alert**: Severe blur or low exposure upgrades the risk category to **Requires Clinical Review**.
- **Low Confidence Alert**: Any model confidence below 50% automatically flags **Requires Clinical Review**.
- **High-Risk Lesion + High Confidence**: Elevates urgency to **Urgent** dermatological evaluation.

---

## 4. Medical Image Analysis Screen & Multi-Session Comparison

- **Interactive 4-Way Comparison View**:
  - Original Dermoscopic Image
  - Preprocessed & Color-Normalized Image
  - ISBI 2016 U-Net Lesion Boundary Segmentation
  - Grad-CAM AI Attention Map & Blended Overlay
- **Multi-Class Probability Distribution**:
  - Full confidence breakdown across all 7 diagnostic classes with entropy uncertainty estimation.
- **Longitudinal / Multi-Session Tracking**:
  - Compare any two scans within a clinical case (Baseline vs. Follow-up).
  - Automatically computes confidence delta, severity trajectory (escalating / stable / de-escalating), and clinical follow-up advice.

---

## 5. API Endpoints

- `GET /api/v1/analysis/status`: Platform and AI service availability status.
- `GET /api/v1/analysis/models`: Benchmark comparison metrics across all 5 architectures.
- `POST /api/v1/analysis/compare`: Longitudinal session comparison between baseline and follow-up images.
- `POST /api/v1/upload/{case_id}?architecture={model_name}`: Full multi-model inference, ISBI U-Net mask generation, Grad-CAM XAI, risk categorization, and prediction persistence.
- `GET /api/v1/upload/images/{image_id}/file`: Authenticated image download/display endpoint.
- `GET /api/v1/cases/{case_id}/history`: Chronological prediction records and image analysis history.

