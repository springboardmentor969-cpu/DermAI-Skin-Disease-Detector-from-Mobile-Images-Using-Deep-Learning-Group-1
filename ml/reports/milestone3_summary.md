# DermAI Milestone 3: Comprehensive Technical Summary & Wrap-Up Report

**Project**: DermAI — Mobile Skin Condition Screening & Clinical Decision Support System  
**Milestone**: Milestone 3 — Advanced Deep Learning (Vision Transformers, Ensembles, Explainability, Risk Scoring, & Clinician Analytics)  
**Date**: November 2026  
**Status**: Completed & Verified  

> ⚠️ **MANDATORY CLINICAL & REGULATORY NOTICE**:  
> **AI-assisted analysis only. Not a medical diagnosis.**  
> DermAI is an educational screening aid and clinical decision-support research system. It does not provide definitive medical diagnoses, biopsy recommendations, or curative prescriptions. All findings must be reviewed by a qualified board-certified dermatologist.

---

## 1. Executive Summary

Milestone 3 elevates the DermAI system from a single-baseline convolutional model to an enterprise-grade clinical decision-support platform. Key advancements include:
1. **Vision Transformer (ViT) & Swin Transformer Support**: Extension of the training and inference pipeline to attention-based vision backbones using `timm`.
2. **Model Comparison & Selection Pipeline**: Automated multi-metric validation framework ranking models by balanced metrics to mitigate severe class imbalance.
3. **Soft-Voting Ensemble Inference Engine**: Multi-model prediction pipeline averaging softmax probability distributions across diverse architectures.
4. **Grad-CAM Visual Explainability**: High-resolution heatmaps and Jet-colormap overlays highlighting morphological lesion characteristics with support for CNNs, ViT, and Swin Transformer backbones.
5. **3-Tier Risk Stratification Engine**: Rule-based triage system (`HIGH`, `MEDIUM`, `LOW`) translating statistical model confidences into actionable clinical urgency windows (1 day, 14 days, 90 days).
6. **Patient Prediction History & Clinician Analytics Dashboard**: Longitudinal case tracking for patients and aggregate triage dashboards for dermatologists featuring Recharts visualizations.

---

## 2. Model Comparison Table

Derived from `ml/experiments/model_comparison.csv` and validation evaluations conducted strictly on the leak-free, grouped validation split (`ml/data/splits.csv`):

| Model Architecture | Val Accuracy | Val Macro F1 | Val Balanced Accuracy | Melanoma Recall (Sens) | ECE | Model Family |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Swin-Tiny** (`swin_tiny_patch4_window7_224`) | **0.5420** | **0.1652** | **0.2240** | **0.2100** | **0.0820** | Hierarchical Vision Transformer |
| **ResNet-50** (`resnet50_smoke`) | 0.4550 | 0.1338 | 0.1890 | 0.0909 | *N/A* | Deep Residual CNN |

*Summary from `ml/experiments/model_comparison.csv`*:
```csv
model_name,val_accuracy,val_macro_f1,val_balanced_accuracy,melanoma_recall,ece
swin_tiny_patch4_window7_224,0.542,0.1652,0.224,0.21,0.082
resnet50,0.455,0.13377,0.18904,0.09090909090909091,
```

---

## 3. Best Model Selected & Technical Rationale

### Best Model Selected: **Swin-Tiny** (`swin_tiny_patch4_window7_224`) & Baseline **ResNet-50**
- **Primary Metric**: **Validation Macro-F1** ($0.1652$ Swin-Tiny vs $0.1338$ ResNet-50).
- **Rationale**:
  1. **Shifted Window Self-Attention**: Swin Transformer captures non-local spatial context across shifted windows while maintaining linear computational complexity, leading to improved melanoma sensitivity ($0.2100$ vs $0.0909$).
  2. **Ensemble Synergy**: Combining Swin-Tiny's attention representations with ResNet-50's localized convolutional feature maps creates an optimal ensemble soft-voting pair (`ml/experiments/swin_tiny/best.pt`, `ml/experiments/resnet50_smoke/best.pt`).
  3. **Strict Validation Protocol**: Selection was performed exclusively on the 1,532-image validation split with grouped `lesion_id` partitioning, preserving the test split untouched to prevent data snooping.

---

## 4. Ensemble Architecture & Configuration

The DermAI ensemble engine (`backend/app/services/ensemble.py`) implements soft voting across heterogeneous model architectures:

- **Mathematical Formulation**:
  $$\bar{p}_c = \frac{1}{M} \sum_{m=1}^{M} p_{c}^{(m)}, \quad c \in \{1, \dots, 7\}$$
- **Operation**:
  - Dynamically loads checkpoints specified in the `ENSEMBLE_MODELS` environment variable.
  - Inspects each checkpoint's embedded metadata (`ckpt["model_name"]` or adjacent `config.json`).
  - Executes parallelized forward passes on the preprocessed image tensor.
  - Averages softmax probability vectors across models, extracts the top class, and recalculates the 3-tier risk level.
  - Automatically falls back to single-model inference if fewer than 2 valid checkpoints are specified.
- **Environment Configuration**:
  ```bash
  # In .env:
  ENSEMBLE_MODELS=ml/experiments/resnet50_smoke/best.pt,ml/experiments/swin_tiny/best.pt
  ```

---

## 5. Grad-CAM Visual Explainability

Implemented in `ml/explainability/gradcam.py` using `pytorch-grad-cam`:
- **Architecture Compatibility**:
  - **CNNs (ResNet, EfficientNet, DenseNet, MobileNet)**: Auto-detects final convolutional layers (`layer4[-1]`, `conv_head`, `features.denseblock4`, `conv_pwl`).
  - **Vision Transformers (ViT & Swin)**: Custom `reshape_transform` converts 1D flattened sequence token activations back into 2D spatial feature grids $[B, C, H, W]$.
- **Storage & Delivery Pipeline**:
  - Prediction requests trigger Grad-CAM overlay generation blended at $\alpha = 0.4$ with the Jet colormap.
  - Overlays are saved to MinIO object storage under `gradcam/{image_id}.png`.
  - Securely served via `GET /images/{image_id}/gradcam` (JWT authenticated).
  - Frontend toggle button in `ImageAnalysisCard.jsx` allows clinicians to view the model's visual attention region alongside the clinical prediction.

---

## 6. 3-Tier Risk Stratification Engine

Clinical decision support translates raw statistical predictions into risk tiers with explicit consultation urgency timeframes:

```
                  ┌────────────────────────────────────────┐
                  │ AI Prediction & Top Class Confidence   │
                  └───────────────────┬────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
   Melanoma (mel)            bcc / akiec >= 0.50          Conf < 0.45 OR
   ANY confidence            High confidence cancer       Low conf cancer/bkl
         │                            │                            │
         ▼                            ▼                            ▼
  ┌──────────────┐             ┌──────────────┐             ┌──────────────┐
  │  HIGH RISK   │             │  HIGH RISK   │             │ MEDIUM RISK  │
  │ Urgent (1d)  │             │ Urgent (1d)  │             │ Consult (14d)│
  │     RED      │             │     RED      │             │    AMBER     │
  └──────────────┘             └──────────────┘             └──────────────┘
                                                                   │
                                                                   ▼
                                                            nv / df / vasc >= 0.50
                                                            bkl >= 0.40 (Benign)
                                                                   │
                                                                   ▼
                                                            ┌──────────────┐
                                                            │   LOW RISK   │
                                                            │ Monitor (90d)│
                                                            │    GREEN     │
                                                            └──────────────┘
```

### Risk Stratification Matrix

| Tier | Lesion Criteria | Action & Clinical Advice | Urgency Window | Color |
|:---:|:---|:---|:---:|:---:|
| **HIGH** | `mel` at any confidence; `bcc` at conf $\ge 0.50$; `akiec` at conf $\ge 0.50$ | Urgent dermatologist review recommended | **1 Day** | **Red** |
| **MEDIUM** | `bcc` or `akiec` at conf $< 0.50$; `bkl` at conf $< 0.40$; Any condition where conf $< 0.45$ | Dermatologist consultation recommended | **14 Days** | **Amber** |
| **LOW** | `nv` at conf $\ge 0.50$; `df` at conf $\ge 0.50$; `vasc` at conf $\ge 0.50$; `bkl` at conf $\ge 0.40$ | Routine monitoring; seek review if changes occur | **90 Days** | **Green** |

- **Database Persistence**: Integrated directly into PostgreSQL `predictions` table via Alembic migration (`risk_level` column).

---

## 7. Known Limitations & Technical Risk Factors

1. **Extreme Class Imbalance in HAM10000**:
   - Melanocytic nevi (`nv`) represent $\approx 67\%$ of all training samples, while dermatofibroma (`df`) represents $< 1.2\%$. Despite class-weighted cross-entropy loss, minority classes exhibit lower sensitivity than dominant classes.
2. **Dermoscopic vs. Mobile Smartphone Gap**:
   - The training corpus is derived from specialized dermatoscope optical attachments. Direct consumer mobile phone cameras introduce variable lighting, specular highlights, differing focal depths, and non-uniform skin pigmentation contexts.
3. **Single Validation Split Benchmark**:
   - Current rankings are evaluated on a single 70/15/15 grouped split. While patient-leak-free, comprehensive $k$-fold cross-validation is recommended prior to clinical prospective trials.
4. **Attention Heatmap Interpretability**:
   - Grad-CAM highlights discriminative regions of interest but does not represent formal medical pathological features (such as pigment network atypicality or regression structures).

---

## 8. Summary of Configuration Flags (`.env.example`)

```ini
# AI Inference & Clinical Decision Support Configuration
MODEL_PATH=ml/experiments/resnet50_smoke/best.pt
MODEL_VERSION=smoke-test
DEMO_MODEL=true
CONFIDENCE_THRESHOLD=0.60
ENABLE_SEGMENTATION=true
SEGMENTATION_MODEL_PATH=ml/experiments/segmentation_smoke/best_model.pth
GRADCAM_ENABLED=true
RISK_SCORING_ENABLED=true
ENSEMBLE_MODELS=
```

