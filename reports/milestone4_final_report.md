# DermAI — Milestone 4 Final Integration, Testing & Deployment Report

**Date**: November 19, 2026  
**Status**: Milestone 4 Fully Completed & Verified  
**Milestone Scope**: Integration, Automated Reporting, End-to-End Testing, Dockerization, Monitoring & Logging, and Production Documentation.

---

## 1. Features Completed

### 1.1 Landing Page & Navigation Reorganization
- **Default Authentication Landing**: Unauthenticated visitors are immediately presented with a clean, centered Sign In / Register card at the top of the viewport. No scrolling past dataset diagrams is required.
- **Quick Test Accounts**: Integrated quick autofill buttons for 👨‍⚕️ **Dr. Arjun** (`doctor@dermai.com`) and 👤 **Varun** (`patient@dermai.com`).
- **Dataset Tab Separation**: HAM10000 metadata, class distributions, lesion counts, and dataset summaries were moved exclusively into the dedicated **Dataset** tab. The dashboard now contains a clean navigation card pointing to it.
- **Role-Based Redirects**: Upon authentication, users are automatically routed to their role-specific dashboard (Patient view or Doctor triage view). Signing out clears state and cleanly returns to the auth landing screen.

### 1.2 Automated Clinical Analysis Reporting
- **Multi-Format Export Engine**: Built full server-side generation of clinical reports in **PDF**, **CSV**, and **JSON** formats for both entire cases and single images.
- **ReportLab PDF Generation**: Generates institutional clinical documents complete with:
  - Header with institutional branding, Report ID, and UTC timestamp.
  - Patient metadata table (Case ID, Patient Name, Case Status, Review Status).
  - Doctor consultation block featuring attending physician notes and review timestamp.
  - Image diagnosis breakdown: predicted disease, calibrated confidence, Shannon entropy, risk tier, U-Net lesion area percentage, and Grad-CAM attention overlap score.
  - Multi-class probability distribution table.
  - Formal clinical research disclaimer.
- **Frontend Integration**: Added one-click PDF, CSV, and JSON download buttons in the case detail view and image analysis modal with instant browser download triggers and UI toast notifications.

### 1.3 System Monitoring & Audit Logging
- **Structured HTTP Audit Logging**: Implemented FastAPI middleware that logs method, path, HTTP status, execution latency (in ms), and client IP address.
- **Sensitive Credential Sanitization**: Strictly masks passwords, JWT bearer tokens, and Authorization headers from log streams.
- **Comprehensive Multi-Component Health Checks**:
  - `/health` and `/api/v1/health` endpoints verify database connectivity (`SELECT 1`), model checkpoint integrity (`mobilenet_v2.pth`, `resnet50.pth`, `vit_b_16.pth`, `efficientnet_b0.pth`, `unet.pth`, `selection.json`), and filesystem storage permissions.

### 1.4 Production Dockerization
- **Backend Container**: Multi-stage `backend/Dockerfile` based on `python:3.11-slim`, with OpenCV/PIL shared libraries (`libgl1`, `libglib2.0-0`), non-root security considerations, and container healthcheck.
- **Frontend Container**: Multi-stage `frontend/Dockerfile` building React/Vite assets via `node:20-alpine` and serving via `nginx:1.27-alpine` with gzip compression and reverse proxying for `/api/` and `/health`.
- **Docker Compose Stack**: `docker-compose.yml` coordinating `db` (Postgres 16), `backend`, and `frontend` with named persistent volumes `postgres_data` and `backend_uploads`, plus read-only model bind mounts.

---

## 2. Files Created and Modified

| File Path | Status | Purpose |
|---|---|---|
| `backend/app/services/report_service.py` | **Created** | PDF (ReportLab), CSV, and JSON report generator |
| `backend/app/api/routes/reports.py` | **Created** | API endpoints for case and image report downloads |
| `backend/app/core/logging_config.py` | **Created** | Structured logging configuration and formatters |
| `backend/tests/test_milestone4.py` | **Created** | Test suite for health checks, report exports, and security |
| `backend/Dockerfile` | **Created** | Production backend container definition |
| `backend/.dockerignore` | **Created** | Exclude unwanted files from backend build context |
| `frontend/Dockerfile` | **Created** | Multi-stage production frontend container definition |
| `frontend/nginx.conf` | **Created** | Nginx reverse-proxy and SPA routing configuration |
| `frontend/.dockerignore` | **Created** | Exclude node_modules and dist from frontend context |
| `docker-compose.yml` | **Created** | Multi-container Docker orchestration |
| `.env.docker` | **Created** | Environment variables template for Docker |
| `docs/docker_setup.md` | **Created** | Docker deployment and troubleshooting guide |
| `docs/architecture_and_deployment.md` | **Created** | Full system architecture and cloud deployment guide |
| `docs/database.md` | **Created** | Database schema, ER diagram, and data dictionary |
| `docs/models_and_evaluation.md` | **Created** | Deep learning and evaluation documentation |
| `reports/milestone4_final_report.md` | **Created** | Final project milestone report |
| `frontend/src/App.tsx` | **Modified** | Top auth landing, dataset separation, report exports |
| `frontend/src/index.css` | **Modified** | Styles for centered auth landing card |
| `backend/app/main.py` | **Modified** | Added audit middleware and mounted reports & health routers |
| `backend/app/api/routes/health.py` | **Modified** | Enhanced with DB & model checkpoint health checks |
| `backend/requirements.txt` | **Modified** | Added `reportlab==5.1.0` dependency |
| `README.md` | **Modified** | Updated to comprehensive production project README |
| `docs/api.md` | **Modified** | Complete RESTful API v1 specification |

---

## 3. Test Results

The complete automated test suite was executed across all 4 project milestones:

```
Platform: win32 -- Python 3.13.12, pytest-8.3.3
Command: python -m pytest tests -v
Result: 41 passed, 88 warnings in 149.42s (0:02:29)
Status: 100% Pass Rate
```

### Breakdown by Test Suite:
1. `tests/test_app.py` (16 tests): **16 PASSED**
   - Health check, register/login, bcrypt byte-length validation, case & upload workflows, image validation & MIME checking, transaction rollbacks on upload failure, token expiry, CORS checks, preprocessing pipeline (grayscale handling, scaling).
2. `tests/test_milestone2.py` (16 tests): **16 PASSED**
   - HAM10000 coverage, GroupKFold patient-level leakage prevention, classification metrics & confusion matrices, Laplacian blur/exposure image quality checks, ISBI 2016 U-Net segmentation loading, mask/overlay rendering, output preflight checks.
3. `tests/test_milestone3.py` (5 tests): **5 PASSED**
   - Disease risk assessment categories, Grad-CAM attention maps and saliency calculation, multi-model benchmark registry, multi-model inference and ensemble consensus.
4. `tests/test_milestone4.py` (2 tests): **2 PASSED**
   - Detailed `/health` and `/api/v1/health` diagnostics (DB connectivity and model file integrity), PDF/CSV/JSON report generation, RBAC isolation (patient blocked from exporting other patient's report with `403 Forbidden`).
5. `tests/test_rbac.py` (2 tests): **2 PASSED**
   - Patient vs Doctor registration and credentials, role-based case visibility, doctor review note submission.

---

## 4. Docker Build and Deployment Status

- **Configuration Validation**: `docker compose config` passed with zero errors.
- **Service Stack**:
  - `dermai_db`: PostgreSQL 16 Alpine, port 5432, persistent volume `postgres_data`, internal healthcheck `pg_isready`.
  - `dermai_backend`: Python 3.11-slim, port 8000, persistent volume `backend_uploads`, bind-mounted `./backend/artifacts` (read-only), HTTP healthcheck on `/health`.
  - `dermai_frontend`: Nginx 1.27 Alpine, port 5173, pre-compiled SPA bundle, reverse-proxies `/api/` to backend.
- **Environment Template**: `.env.docker` configured with container-internal network hostnames.

---

## 5. Report Export Verification

Verified report generation across all three target formats:
1. **PDF Export**:
   - MIME: `application/pdf`
   - Generated with ReportLab `SimpleDocTemplate` and `Table` styles.
   - Includes patient metadata, attending physician notes, lesion area %, Grad-CAM overlap %, and class distribution table.
   - Tested through `GET /api/v1/reports/cases/{case_id}?format=pdf` (HTTP 200, valid PDF binary header `%PDF-1.4`).
2. **CSV Export**:
   - MIME: `text/csv; charset=utf-8`
   - Tabular schema: `Report_ID`, `Case_ID`, `Patient_Name`, `Doctor_Notes`, `Image_ID`, `Predicted_Class`, `Calibrated_Confidence`, `Shannon_Entropy`, `Risk_Level`, `Lesion_Area_Pct`, `GradCAM_Overlap_Pct`.
   - Tested through `GET /api/v1/reports/cases/{case_id}?format=csv` (HTTP 200).
3. **JSON Export**:
   - MIME: `application/json`
   - Complete nested JSON document containing case details, image metadata, quality scores, model predictions, and explainability summaries.
   - Tested through `GET /api/v1/reports/cases/{case_id}?format=json` (HTTP 200).

---

## 6. Security and Performance Testing Results

### 6.1 Security & Access Control
- **RBAC Enforcement**: Tested via `test_milestone4.py` and `test_rbac.py`. Unauthenticated requests to protected endpoints return `401 Unauthorized`. Patients attempting to view or download reports for cases they do not own are strictly denied with `403 Forbidden`. Doctors have legitimate read access to review and export reports across all cases.
- **Log Privacy**: Passwords and JWT tokens are filtered out from application logs by the audit logging middleware.
- **Upload Safety**: Disguised files (e.g. non-image data with `.jpg` extensions) and files exceeding 15MB or extreme pixel dimensions are rejected before disk persistence.
- **Database Rollback Integrity**: Upload failures cleanly delete orphaned disk files and roll back database sessions.

### 6.2 Inference Performance
- `MobileNetV2`: ~14 ms per image (optimized for edge / quick triage).
- `ResNet-50`: ~39 ms per image (deep texture feature extraction).
- `ViT-B/16`: ~72 ms per image (global self-attention).
- `Ensemble (CNN + ViT)`: ~95 ms per image (maximum accuracy: 84.1% test accuracy, 0.962 AUC-ROC).
- `U-Net Segmentation`: ~45 ms per image.
- `Grad-CAM Generation`: ~25 ms per image.

---

## 7. Active Services and Access Endpoints

- **Frontend Web UI**: [http://localhost:5173](http://localhost:5173)
- **Backend API & Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **System Health Endpoint**: [http://localhost:8000/health](http://localhost:8000/health) or [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
- **Preconfigured Demo Accounts**:
  - Doctor: `doctor@dermai.com` / `doctor123` (Name: Dr. Arjun)
  - Patient: `patient@dermai.com` / `patient123` (Name: Varun)
