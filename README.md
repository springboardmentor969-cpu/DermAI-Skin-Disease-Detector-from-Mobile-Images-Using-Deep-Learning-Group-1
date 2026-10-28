# DermAI: Skin Disease Detector from Mobile Images Using Deep Learning
## Milestone 1 (Weeks 1 & 2) — Project Initialization, Dataset & Core Setup

> **Clinical Decision-Support Disclaimer:** DermAI is designed strictly as an AI-assisted dermatological screening, research, and clinical decision-support system. It is **not** an autonomous medical diagnostic system.

---

### Milestone 1 Completed Deliverables
1. **Project Objectives & End-to-End Workflow (`docs/PROJECT_SPEC.md`)**: Complete definition of screening workflow (`Mobile Camera -> Image Capture -> Quality Check -> User Confirmation -> Upload -> Preprocessing -> AI Ready`).
2. **Target Skin Conditions & Public/De-Identified Datasets (`ml_pipeline/dataset_config.py`)**:
   - **HAM10000** (10,015 dermoscopic images across 7 classes: `nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`)
   - **PAD-UFES-20** (2,298 smartphone-captured clinical images across 6 classes: `BCC`, `SCC`, `ACK`, `SEK`, `MEL`, `NEV`)
3. **Exploratory Data Analysis & Class Distribution (`ml_pipeline/eda_analysis.py`)**: Computes class frequencies, imbalance ratios, inverse-frequency class weights for Focal/Weighted Cross-Entropy Loss, anatomical site breakdown, and patient age/sex distributions.
4. **System Architecture & PostgreSQL Database Schema (`backend/schema.sql`, `backend/app/models.py`)**:
   - 5-Layer Microservices Architecture (Client Layer, API Gateway/Auth Layer, Application Services Layer, Data Layer, Infrastructure Layer).
   - Relational schema for `users`, `patients` (de-identified), `cases`, `case_images`, `preprocessing_logs`, and `audit_logs`.
5. **Authentication & Role-Based Access Control (`backend/app/auth.py`)**:
   - JWT Bearer authentication, OAuth2 password flow, BCrypt password hashing, session management, and RBAC across all 5 required roles:
     - `Patient / General User`
     - `Dermatologist / Clinician`
     - `Medical Researcher`
     - `Data Scientist / ML Engineer`
     - `Administrator`
6. **Patient & Case Management (`backend/app/routers/cases.py`)**:
   - Create de-identified patient records and skin-lesion cases, attach clinical notes, track body location, image source, model version, and analysis status.
7. **Mobile Image Capture, Validation & Quality Check (`backend/app/services/preprocessing.py`)**:
   - Validates format (`JPEG/PNG/WEBP`), resolution (`>=224x224`), file size, blur level (Laplacian variance), brightness, contrast, exposure, and lesion visibility (`Good Quality`, `Acceptable Quality`, `Poor Quality`).
8. **Initial Image Preprocessing Engine (`backend/app/services/preprocessing.py` & `ml_pipeline/preprocessing_pipeline.py`)**:
   - **Step 1:** ROI Cropping & Orientation Correction
   - **Step 2:** Standardized Resizing (`224x224`)
   - **Step 3:** Hair & Artifact Removal (Morphological Black-Hat DullRazor algorithm + Telea Inpainting)
   - **Step 4:** Contrast Enhancement (LAB-space CLAHE)
   - **Step 5:** Color Normalization (Shades-of-Gray / Gray-World illumination constancy)
   - **Step 6:** Pixel Normalization (`[0, 1]` scaling & ImageNet `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

---

### Quick Start (Local Development)

#### Option A: Run with Docker Compose (Recommended for Hosting)
```bash
docker compose up --build
```
- **Frontend Web App:** `http://localhost:5173`
- **FastAPI Backend & Swagger UI:** `http://localhost:8000/docs`
- **PostgreSQL Database:** `localhost:5432`

#### Option B: Run Manually (Python + Node.js)

1. **Start the FastAPI Backend (defaults to local SQLite if `DATABASE_URL` is unset):**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Run Exploratory Data Analysis & Preprocessing CLI:**
```bash
cd ml_pipeline
python3 eda_analysis.py --dataset ham10000 --output eda_report.json
python3 preprocessing_pipeline.py --input ../sample_data/images/sample_02_melanoma_hair_artifact.png --output ../sample_data/processed_sample.png
```

3. **Start the React + Tailwind Frontend:**
```bash
cd frontend
npm install
npm run dev
```
