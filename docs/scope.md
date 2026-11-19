# Project Scope & Operational Workflow

> ⚠️ **MEDICAL DISCLAIMER**:  
> **DermAI is an educational screening and clinical decision-support tool — NOT a definitive medical diagnosis.**  
> It is designed to assist clinical triage and promote early risk stratification. It does not replace the professional clinical judgment, physical examination, dermoscopic evaluation, or histopathological biopsy by a licensed physician or board-certified dermatologist.

---

## 1. Project Objective

Skin cancers—including melanoma, basal cell carcinoma, and squamous cell carcinoma—are among the most commonly diagnosed cancers globally. Early detection is the single most critical factor in successful clinical outcomes and patient survival. 

The primary objective of **DermAI** is to provide an accessible, intelligent, mobile-first screening and decision-support platform that:
1. **Empowers Patients**: Enables individuals to securely photograph, track, and monitor suspicious skin lesions.
2. **Assists Clinicians**: Equips general practitioners and dermatologists with automated pre-screening risk scores, lesion categorization, and centralized case queues.
3. **Ensures Data Integrity**: Validates photograph quality (blur, lighting, contrast, resolution) before running neural network inference, preventing misleading predictions caused by poor imaging.
4. **Protects Patient Privacy**: Implements strict architectural separation between identifiable user accounts and de-identified clinical case records.

---

## 2. Target Users & Stakeholders

| User Role | Description & Primary Capabilities |
|:---|:---|
| **Patient** | • Creates screening cases with non-identifying demographics (age, gender, body site).<br>• Captures or uploads lesion photos using mobile cameras or photo gallery.<br>• Receives real-time image quality feedback (sharpness, lighting).<br>• Accesses educational risk stratification and strictly views their own cases. |
| **Clinician** | • Accesses all screening cases across the repository for triage.<br>• Reviews high-resolution dermoscopic photographs and automated quality metrics.<br>• Observes deep learning top-N predictions and model confidence scores.<br>• Documents clinical observations and recommends in-person biopsy or follow-up. |
| **Researcher** | • Explores anonymized datasets and diagnostic correlations across age, sex, and anatomical sites.<br>• Audits class distributions and evaluates algorithm performance across cohorts. |
| **ML Engineer** | • Manages model versions, benchmark datasets (HAM10000), and training pipelines.<br>• Validates preprocessing pipelines (EXIF correction, center-cropping, normalization).<br>• Monitors prediction confidence thresholds and quality rejection rates. |
| **Administrator** | • Oversees system health, audit logs, database integrity, and storage governance.<br>• Admin registration is restricted internally to prevent unauthorized privilege escalation. |

---

## 3. End-to-End Operational Workflow

The DermAI screening journey follows a rigorous 5-stage pipeline to ensure clinical safety and imaging reliability:

```
┌─────────────┐     ┌───────────────────┐     ┌──────────────────────┐     ┌───────────────┐     ┌──────────────────┐
│  1. CAPTURE │ ──> │ 2. QUALITY CHECK  │ ──> │   3. CONFIRMATION    │ ──> │   4. UPLOAD   │ ──> │  5. AI ANALYSIS  │
│ Camera/File │     │ Blur / Light / Res│     │ Review & Crop Framing│     │ MinIO Storage │     │ PyTorch Inference│
└─────────────┘     └───────────────────┘     └──────────────────────┘     └───────────────┘     └──────────────────┘
```

### Stage 1: Capture
- **Action**: The user initiates a screening case by specifying the anatomical body location (e.g., *Left Forearm*, *Upper Back*) and optional non-identifying patient demographics (age, gender, clinical notes).
- **Imaging**: The user either activates their device camera directly (`capture="environment"`) or selects an existing high-resolution photograph from their photo gallery.
- **Accepted Formats**: Standard JPEG or PNG files up to 10 MB in file size.

### Stage 2: Quality Check (Pre-flight Validation)
- Prior to model evaluation, the image undergoes automated computer vision assessment (`ml/preprocessing/quality.py`):
  1. **Integrity Validation**: Verifies genuine binary image structure using Pillow (rejecting corrupted files or renamed non-images).
  2. **Resolution Check**: Confirms image meets the minimum dimension threshold (at least 224 × 224 pixels).
  3. **Blur Score**: Calculates the Laplacian variance ($\sigma^2_{\Delta}$). Scores $< 60.0$ indicate unacceptable motion blur or focal defects.
  4. **Brightness Assessment**: Computes mean grayscale pixel intensity ($0 \dots 255$). Detects severe under-exposure ($< 45.0$) or over-exposure/flash washout ($> 220.0$).
  5. **Contrast Analysis**: Measures RMS intensity standard deviation. Scores $< 25.0$ identify flat, low-contrast images lacking visible pigment structures.
- **Categorization**: Classifies the photo into **`good`**, **`acceptable`**, or **`poor`**.

### Stage 3: Confirmation & Framing
- **Framing**: The user previews the photograph and uses the interactive 1:1 square crop tool (`react-image-crop`) to center the lesion and eliminate extraneous background clutter.
- **Quality Alert**:
  - If the photograph is classified as **`poor`**, the system highlights the specific imaging defects (e.g., *blurry photo*, *under-exposed*) and displays an alert requesting the user to retake the photo in bright, uniform lighting.
  - To prevent data loss in remote telemedicine workflows, poor photos can still be preserved in the case log, but they are visibly flagged as sub-optimal.

### Stage 4: Secure Upload & Persistence
- **Object Storage**: The cropped, verified photograph is uploaded to MinIO (S3-compatible object storage) under an unguessable unique key: `cases/{case_id}/{uuid}.jpg`.
- **Database Indexing**: Metadata is persisted in the PostgreSQL `images` table:
  - `storage_key`, `source` (camera vs gallery), `captured_at`, `quality_label`, and detailed quality metrics.
- **Audit Trail**: An immutable record is created in the `audit_logs` table (`IMAGE_UPLOADED`), capturing the timestamp, acting user ID, and image properties.

### Stage 5: AI Analysis (Decision Support)
- **Preprocessing Pipeline** (`ml/preprocessing/preprocess.py`):
  1. Reads EXIF tags and applies orientation correction (`exif_transpose`).
  2. Crops to symmetrical square aspect ratio.
  3. Resizes via bicubic interpolation to the neural network input size (224 × 224).
  4. Converts to standardized 3-channel RGB.
  5. Normalizes pixel values with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
  6. Formats into a PyTorch-ready contiguous tensor of shape `(3, 224, 224)`.
- **Inference**: The model evaluates morphological lesion features, outputting predicted diagnosis, confidence percentage, and top-$N$ class probabilities for clinician review.

---

## 4. Ethical & Clinical Boundaries

1. **Non-Diagnostic Nature**: DermAI does not provide a definitive diagnosis. It serves solely as an early-warning risk triage mechanism.
2. **Physician-in-the-Loop**: Automated assessments are accompanied by clear disclaimers directing high-risk cases to qualified medical professionals.
3. **Privacy First**: Patient identifiable information (names, national IDs, addresses) is never stored alongside lesion photographs.
