# DermAI: Mobile Skin Condition Screening & Clinical Decision Support System

[![Backend Tests](https://img.shields.io/badge/pytest-24%20passed-emerald)](backend/tests/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.3-61dafb?logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-8.3-646cff?logo=vite&logoColor=white)](https://vitejs.dev)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-v4-38bdf8?logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169e1?logo=postgresql&logoColor=white)](https://www.postgresql.org)
[![MinIO](https://img.shields.io/badge/MinIO-S3_Compatible-c72c48?logo=minio&logoColor=white)](https://min.io)

> ⚠️ **MEDICAL & CLINICAL DISCLAIMER**:  
> **DermAI is an educational screening and clinical decision-support tool — NOT a definitive medical diagnosis.**  
> It is designed to assist clinical triage and promote early risk stratification. It does not replace professional clinical judgment, physical examination, dermoscopic evaluation, or histopathological biopsy by a licensed physician or board-certified dermatologist.

---

## 📌 Project Overview

**DermAI** is an end-to-end, privacy-preserving mobile dermatology screening and decision-support system. It enables patients and healthcare providers to capture skin lesion photographs, assess visual image quality in real time, store scans securely in S3-compatible object storage, segment lesion boundaries, and perform multi-class disease classification with uncertainty awareness.

### System Capabilities

#### Milestone 1 (Foundation & Storage)
- **Role-Based Authentication**: Secure JWT authentication with Bcrypt password hashing and role enforcement (`patient`, `clinician`, `researcher`, `ml_engineer`, `admin`).
- **Clinical Case Management**: Non-identifying demographic logging (`patients`) isolated from user account credentials (`users`).
- **Intelligent Pre-Flight Quality Check**: Automated computer vision inspection measuring blur (Laplacian variance), illumination, contrast, and resolution before deep learning inference.
- **Interactive Cropping**: 1:1 square lesion framing via `react-image-crop` to optimize input geometry for convolutional neural networks.
- **Secure Object Storage**: Automated S3 upload to MinIO under unguessable UUID storage keys (`cases/{id}/{uuid}.jpg`).
- **Comprehensive Audit Trail**: Tamper-evident logging of every login, case creation, upload, and analysis request in PostgreSQL.
- **Leak-Free ML Dataset Split**: Stratified 70/15/15 split grouped strictly by `lesion_id` on the 10,015 HAM10000 dataset to prevent feature memorization.

#### Milestone 2 (Deep Learning, Segmentation & Inference API)
- **Lesion Boundary Segmentation**: U-Net with a pre-trained ResNet-34 encoder (`segmentation_models_pytorch`) trained on ISIC 2018 Task 1 data using combined Dice + Binary Cross-Entropy loss. Extracts lesion masks, bounding boxes, and cropped ROIs with safety margins.
- **Multi-Class Skin Disease Classification**: Support for modern deep learning architectures (**ResNet-50**, **EfficientNet-B0**, **DenseNet-121**, **MobileNetV3-Large**) with two-phase transfer learning and class-weighted/focal loss.
- **Dynamic Model Loading Service**: `backend/app/services/inference.py` loads model architectures, weights, and class mappings dynamically from `config.json` or `.pt` checkpoints without code modifications.
- **FastAPI AI Analysis Endpoints**:
  - `POST /images/{image_id}/analyze`: Verifies user case access, validates quality (rejecting poor images), performs inference, formats top-3 candidate predictions, enforces confidence thresholds, logs audit events, and saves predictions to the database.
  - `GET /images/{image_id}/predictions`: Retrieves historical prediction runs for an image.
- **Confidence Calibration & Selective Triage**: Temperature scaling optimization and threshold sweeps ($\tau \in [0.40, 0.90]$) via `ml/evaluation/calibrate_confidence.py` to route uncertain cases to mandatory clinical review.
- **Standardized Multi-Model Evaluation**: Automated evaluation suite (`evaluate.py`, `compare_models.py`) generating confusion matrices, ROC curves, macro-F1 bar charts, and markdown comparison tables strictly on the validation split.
- **Frontend Clinical Decision Card**: Calm blue UI presentation displaying top prediction, confidence gauge, top-3 candidates, demo model banners, and mandatory disclaimers.

---

## 📁 Repository Structure

```text
DermAI/
├── backend/                  # FastAPI REST API Service
│   ├── alembic/              # Alembic database migration environment
│   │   └── versions/         # Migration scripts
│   ├── app/
│   │   ├── main.py           # Application entrypoint & lifespan
│   │   ├── config.py         # Pydantic Settings configuration from .env
│   │   ├── database.py       # SQLAlchemy engine & session management
│   │   ├── dependencies.py   # JWT get_current_user & require_role dependencies
│   │   ├── models/           # Models (User, Patient, Case, Image, Prediction, AuditLog)
│   │   ├── schemas/          # Pydantic validation & response schemas
│   │   ├── routers/          # Routes (health, auth, users, cases, images)
│   │   └── services/         # Business logic (auth_service, storage_service, inference)
│   ├── tests/                # Automated pytest suite (24 passing unit/integration tests)
│   ├── alembic.ini           # Alembic configuration
│   └── requirements.txt      # Python dependencies
├── frontend/                 # React 19 + Vite + Tailwind CSS Web Application
│   ├── src/
│   │   ├── api/              # Axios API client with Bearer token interceptor
│   │   ├── components/       # ImageAnalysisCard, Navbar, Footer, ProtectedRoute, QualityBadge
│   │   ├── context/          # AuthContext for session management
│   │   ├── pages/            # LoginPage, RegisterPage, DashboardPage, CaseDetailPage, UploadPage
│   │   ├── App.jsx           # Routing & layout
│   │   └── index.css         # Tailwind v4 styling
│   ├── vite.config.js        # Vite config with React & Tailwind plugins
│   └── package.json          # Node dependencies
├── ml/                       # Machine Learning Subsystem
│   ├── data/
│   │   ├── raw/              # HAM10000 images & metadata
│   │   ├── isic2018_task1/   # ISIC 2018 Task 1 images & masks
│   │   └── splits.csv        # Leak-free grouped 70/15/15 splits
│   ├── preprocessing/        # quality.py (quality metrics) & preprocess.py (PyTorch pipeline)
│   ├── training/             # train.py, dataset.py, augmentations.py, check_data.py
│   ├── segmentation/         # model.py, dataset.py, losses.py, metrics.py, train.py, segment.py
│   ├── evaluation/           # evaluate.py, compare_models.py, calibrate_confidence.py
│   ├── experiments/          # Model checkpoints, metrics, plots, and comparison reports
│   └── notebooks/            # 01_eda.ipynb (exploratory data analysis)
├── docs/                     # Project Architecture & Milestone Reports
│   ├── scope.md              # Project scope, personas, and workflows
│   ├── architecture.md       # Architecture diagram & component specifications
│   ├── database.md           # Mermaid ER diagram & privacy architecture
│   ├── milestone1-checklist.md # Milestone 1 evaluation criteria checklist
│   └── milestone2.md         # Milestone 2 comprehensive evaluation & technical report
├── docker-compose.yml        # PostgreSQL 16 & MinIO S3 services
├── pytest.ini                # Pytest configuration
├── test_preprocessing.py     # Preprocessing evaluation script
├── .env.example              # Environment variables template
├── .env                      # Local configuration (ignored by Git)
└── README.md                 # Project README
```

---

## ⚙️ Environment Variables

All configuration is centralized in `.env` and loaded by FastAPI and the frontend:

| Variable | Default Value | Description |
|:---|:---|:---|
| `ENVIRONMENT` | `development` | Deployment environment (`development` / `production`). |
| `POSTGRES_USER` | `dermai_user` | PostgreSQL database username. |
| `POSTGRES_PASSWORD` | `dermai_password` | PostgreSQL database password. |
| `POSTGRES_DB` | `dermai_db` | PostgreSQL database name. |
| `DATABASE_URL` | `postgresql://...` | SQLAlchemy connection URI. |
| `MINIO_ROOT_USER` | `minioadmin` | MinIO administrative access key. |
| `MINIO_ROOT_PASSWORD`| `minioadmin` | MinIO administrative secret key. |
| `MINIO_ENDPOINT` | `localhost:9000` | S3 client endpoint used by Boto3. |
| `MINIO_BUCKET_NAME` | `dermai-images` | Target bucket for photograph storage. |
| `SECRET_KEY` | `supersecretkey_...` | Secret key used to sign JWT Bearer tokens. |
| `ALGORITHM` | `HS256` | JWT signature algorithm. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | JWT token validity window in minutes. |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Base URL used by the frontend Axios client. |
| `MODEL_PATH` | `ml/experiments/resnet50_smoke/best.pt` | Path to active classification model checkpoint. |
| `MODEL_VERSION` | `smoke-test` | Model version identifier string reported in API responses. |
| `DEMO_MODEL` | `true` | When true, renders demo banner indicating non-clinical weights. |
| `CONFIDENCE_THRESHOLD` | `0.60` | Operating threshold below which cases are flagged for clinical review. |
| `ENABLE_SEGMENTATION` | `true` | Enables lesion localization and ROI cropping during analysis. |
| `SEGMENTATION_MODEL_PATH` | `ml/experiments/segmentation_smoke/best_model.pth` | U-Net segmentation model checkpoint. |

---

## Quick Start (Development)
```bash
cp .env.example .env   # fill in your values
docker compose up -d
```

## Production
```bash
docker compose -f docker-compose.prod.yml up -d
```

## Running Tests
```bash
make test
```

---

## 🚀 Detailed Setup Guide

### 1. Start Infrastructure Containers (Docker Compose)
Start the PostgreSQL database and MinIO object storage:
```powershell
docker compose up -d
```
Verify running services:
- **PostgreSQL**: `localhost:5432`
- **MinIO Console**: [http://localhost:9001](http://localhost:9001) (`minioadmin` / `minioadmin`)

---

### 2. Backend Setup & Server Execution

1. In a terminal at the project root, install Python dependencies:
   ```powershell
   pip install -r backend/requirements.txt
   ```

2. Run database migrations:
   ```powershell
   cd backend
   alembic upgrade head
   cd ..
   ```

3. Launch the FastAPI server:
   ```powershell
   uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   - **Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

### 3. Frontend Web App Setup

1. Open a new terminal in `frontend/` and install dependencies:
   ```powershell
   cd frontend
   npm install
   ```

2. Start the Vite development server:
   ```powershell
   npm run dev
   ```
   - Access the UI at [http://localhost:5173](http://localhost:5173).

---

## 🔬 Machine Learning Subsystem CLI Guide

### 1. Data Verification & Dataset Split Audit
Verify image integrity and audit the leak-free patient-grouped split:
```powershell
python -m ml.training.check_data
```

### 2. Classification Model Training
Train a supported backbone (`resnet50`, `efficientnet_b0`, `densenet121`, `mobilenet_v3_large`) using class-weighted cross-entropy:
```powershell
python -m ml.training.train --model_name resnet50 --loss ce_weighted --batch_size 32 --epochs_freeze 3 --epochs_tune 15 --exp_name resnet50_weighted
```

### 3. Model Evaluation on Validation/Test Splits
Evaluate a trained model checkpoint and generate confusion matrices and ROC curves:
```powershell
python -m ml.evaluation.evaluate --checkpoint ml/experiments/resnet50_smoke/best.pt --split val --output_dir ml/experiments/resnet50_smoke/eval_val
```

### 4. Automated Multi-Model Comparison & Ranking
Evaluate all models in `ml/experiments/` on the validation set, select the best model by validation macro-F1, and create comparison charts:
```powershell
python -m ml.evaluation.compare_models --experiments_dir ml/experiments
```
> [!IMPORTANT]
> The test split must **never** be used for model selection. Only after finalizing the best model on validation, evaluate once on the test split using:
> ```powershell
> python -m ml.evaluation.compare_models --run_test --test_model resnet50
> ```

### 5. Confidence Calibration & Operating Point Sweep
Fit temperature scaling on the validation split and inspect coverage vs. recall trade-offs:
```powershell
python -m ml.evaluation.calibrate_confidence --checkpoint ml/experiments/resnet50_smoke/best.pt --output_dir ml/experiments/confidence_calibration
```

### 6. Lesion Segmentation Training
Train the U-Net ResNet-34 segmentation network on ISIC 2018 Task 1 data:
```powershell
python -m ml.segmentation.train --data_dir ml/data/isic2018_task1 --epochs 10 --batch_size 16 --output_dir ml/experiments/segmentation_run1
```

---

## 🧪 Automated Testing Suite

The repository includes a comprehensive automated test suite spanning authentication, case management, and AI inference:

```powershell
# Run all backend unit and integration tests
pytest

# Run image quality evaluation benchmark on sample images
python ml/preprocessing/test_preprocessing.py

# Verify frontend production build
npm --prefix frontend run build
```

### Verified Test Coverage (24 Passing Tests)
- **Authentication & RBAC (`backend/tests/test_auth.py`)**: 8 tests verifying registration, duplicate email rejection, admin creation lockdown, login tokens, failed attempts, and role protection.
- **Case & Image Management (`backend/tests/test_cases.py`)**: 8 tests verifying case creation, patient demographic isolation, role-based visibility, file type and size validation, image uploads, and image streaming.
- **AI Inference & Quality Gating (`backend/tests/test_inference.py`)**: 8 tests verifying dynamic config loading, quality rejection (rejecting "poor" images with 400), successful analysis, prediction persistence, confidence threshold warnings, top-3 candidates, demo model flags, and mandatory disclaimers.

---

## 📚 Project Documentation Links

- **[Milestone 2 Technical Report](docs/milestone2.md)**: Comprehensive review of Milestone 2 criteria, models trained, comparison table, segmentation results, melanoma confusion analysis, confidence calibration, and known limitations.
- **[Milestone 1 Checklist](docs/milestone1-checklist.md)**: Milestone 1 deliverables and evaluation checklist.
- **[System Architecture](docs/architecture.md)**: Mermaid system architecture diagram, component interactions, and security protocols.
- **[Database Schema & ER Diagram](docs/database.md)**: Mermaid ER diagram, data dictionaries, and HIPAA/GDPR privacy design.
- **[Project Scope & Workflow](docs/scope.md)**: Clinical user personas, operational screening workflows, and disclaimers.
