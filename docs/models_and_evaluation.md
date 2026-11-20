# DermAI — Model Architecture, Training & Evaluation Documentation

## 1. Datasets & Data Hygiene

### 1.1 HAM10000 Dataset (Classification & Diagnosis)
- **Total Images**: 10,015 dermoscopic photographs across 7 diagnostic categories:
  1. `mel`: Melanoma (1,113 lesions)
  2. `nv`: Melanocytic Nevi (6,705 lesions)
  3. `bkl`: Benign Keratosis-like lesions (1,099 lesions)
  4. `bcc`: Basal Cell Carcinoma (514 lesions)
  5. `akiec`: Actinic Keratoses / Intraepithelial Carcinoma (327 lesions)
  6. `vasc`: Vascular Lesions (142 lesions)
  7. `df`: Dermatofibroma (115 lesions)
- **Data Leakage Safeguard**: Lesions with multiple follow-up captures are partitioned strictly at the `lesion_id` level using GroupKFold. Images of the same patient lesion never span train and test sets.

### 1.2 ISBI 2016 Part 1 Dataset (Lesion Segmentation)
- Independent dataset with ground-truth binary lesion boundaries. HAM10000 classification labels are strictly never repurposed as pixel masks.
- Used exclusively to train and evaluate the U-Net segmentation network.

---

## 2. Model Architectures & Benchmark Comparison

DermAI evaluates five distinct model configurations to balance diagnostic accuracy, parameter efficiency, and runtime latency:

| Model ID | Architecture | Parameter Count | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | AUC-ROC | Inference Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `mobilenet_v2` | Inverted Residuals & Linear Bottlenecks | 3.5M | 77.2% | 0.595 | 77.0% | 0.565 | 0.912 | 14 ms |
| `efficientnet_b0` | Compound Scaling (Depth/Width/Res) | 5.3M | 75.1% | 0.572 | 74.8% | 0.549 | 0.908 | 23 ms |
| `resnet50` | Residual Identity Skip Connections | 25.6M | 81.4% | 0.658 | 81.0% | 0.642 | 0.938 | 39 ms |
| `vit_b_16` | Pure Vision Transformer (16x16 Patches) | 86.6M | 82.9% | 0.684 | 82.4% | 0.671 | 0.949 | 72 ms |
| **`ensemble`** | **Soft-Voting Ensemble (CNN + ViT)** | **115.7M** | **84.6%** | **0.712** | **84.1%** | **0.698** | **0.962** | **95 ms** |

---

## 3. Lesion Segmentation (U-Net)

- **Architecture**: 4-level contracting encoder path with double convolutions ($3 \times 3$, ReLU, BatchNorm) and max-pooling ($2 \times 2$), paired with expansive decoder path with transposed convolutions and skip-concatenations.
- **Output**: Single-channel Sigmoid probability map thresholded at $\tau = 0.50$ to produce binary boundary mask.
- **Postprocessing**: Morphological hole-filling, largest connected component filtering, and smoothed contour blending.

---

## 4. Explainable AI (XAI) & Grad-CAM Pipeline

- **Convolutional Backbones**:
  - Hooks gradients $\nabla_{A^k} y^c$ and feature activations $A^k$ at the terminal convolutional stage (`layer4` for ResNet-50, `features[-1]` for MobileNetV2 / EfficientNet-B0).
  - Calculates global average pooled weights $\alpha_k^c$ and generates activation map $L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$.
- **Vision Transformer Backbone**:
  - Computes gradient-weighted spatial saliency from CLS token attention over 196 patch tokens.
- **Visualization Artifacts**:
  - `heatmap_data_url`: Base64 PNG with JET colormap normalization.
  - `overlay_data_url`: Alpha-blended composite ($0.6 \times \text{RGB} + 0.4 \times \text{Heatmap}$).
  - `attention_overlap_pct`: Intersection over lesion area indicating model focus validity.

---

## 5. Clinical Safety & Risk Matrix

Predictions are mapped to clinical urgency tiers:
- **High Risk**: Melanoma (`mel`, Severity 92/100) and Basal Cell Carcinoma (`bcc`, Severity 80/100).
- **Moderate Risk**: Actinic Keratosis (`akiec`, Severity 62/100).
- **Low Risk / Benign**: Benign Keratosis (`bkl`), Melanocytic Nevi (`nv`), Dermatofibroma (`df`), Vascular Lesion (`vasc`).

**Safety Overrides**:
- Blur score $< 100$ or extreme underexposure flags **Requires Clinical Review**.
- Shannon entropy $> 1.5$ nats or top confidence $< 50\%$ flags **Requires Clinical Review**.
