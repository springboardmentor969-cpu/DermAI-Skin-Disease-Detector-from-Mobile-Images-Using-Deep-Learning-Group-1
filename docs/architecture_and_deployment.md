# DermAI — System Architecture & Deployment Guide

## 1. System Architecture Overview

DermAI is an end-to-end, privacy-conscious dermatological decision support platform. It integrates deep neural networks, explainable AI (XAI), lesion segmentation, and clinical role-based workflows into a unified modern web application.

```
+-------------------------------------------------------------------------------+
|                                CLIENT TIER                                    |
|   React 18 + TypeScript + Vite + Tailwind CSS SPA                             |
|   - Clean Landing Page / Authentication Modal (Doctor vs. Patient Roles)     |
|   - Multi-Session Case Tracker & Longitudinal Comparison                      |
|   - 4-Way Diagnostic Image Studio (Original / Preprocessed / U-Net / Grad-CAM)|
|   - HAM10000 Dataset Inspection & Statistics Hub                              |
+---------------------------------------+---------------------------------------+
                                        | HTTP / JSON / File Multipart
                                        v
+-------------------------------------------------------------------------------+
|                       API GATEWAY & REVERSE PROXY                             |
|   Nginx 1.27 Reverse Proxy (Containerized) / Local Vite Dev Proxy             |
|   - Static SPA Asset Delivery (Gzip enabled, cache-control)                   |
|   - /api/v1/ and /health Passthrough Routing                                  |
+---------------------------------------+---------------------------------------+
                                        | Internal Reverse Proxy
                                        v
+-------------------------------------------------------------------------------+
|                           APPLICATION SERVER (FASTAPI)                        |
|   FastAPI + Uvicorn Async Server                                              |
|   - Security & RBAC Middleware (JWT Bearer Token, Bcrypt 12 rounds)           |
|   - Structured Audit Logging (Request latency, IP, credential masking)        |
|   - Modular Routers: Auth, Cases, Upload, Analysis, Reports, Health, Dataset  |
+---------+-------------------+-------------------+-------------------+---------+
          |                   |                   |                   |
          v                   v                   v                   v
+-------------------+ +-------------------+ +-------------------+ +-------------------+
|  POSTGRESQL 16    | | DEEP LEARNING     | | EXPLAINABILITY    | | REPORT ENGINE     |
|  - Users (RBAC)   | | - MobileNetV2     | | - Grad-CAM        | | - ReportLab PDF   |
|  - Cases          | | - ResNet-50       | | - Patch Saliency  | | - Tabular CSV     |
|  - Images         | | - ViT-B/16        | | - JET Colormap    | | - Raw JSON        |
|  - Predictions    | | - EfficientNet-B0 | | - Lesion Overlap  | |                   |
|  - Doctor Reviews | | - Soft Ensemble   | |                   | |                   |
+-------------------+ | - ISBI U-Net      | +-------------------+ +-------------------+
                      +-------------------+
```

---

## 2. Core Subsystems

### 2.1 Role-Based Access Control (RBAC)
The platform enforces strict role separation:
- **Patient Role (`user`)**:
  - Can register and sign in.
  - Can create personal cases and upload skin lesion photographs.
  - Can view their own cases, image analyses, and generated clinical reports.
  - Denied access to other patients' cases (returns `403 Forbidden`).
  - Cannot submit clinical review notes or override diagnoses.
- **Doctor / Dermatologist Role (`doctor`)**:
  - Has global access to all submitted clinical cases across all patients.
  - Can inspect uploaded dermoscopy scans, U-Net lesion masks, and Grad-CAM attention maps.
  - Can submit official clinical consultation notes and mark cases as `reviewed`.
  - Can export comprehensive multi-format clinical reports for patient records.

### 2.2 Deep Learning & Inference Pipeline
1. **Input Preprocessing**:
   - MIME validation and PIL image decoding.
   - Resize to $224 \times 224$ pixels.
   - Normalization using ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
2. **Multi-Model Inference Engine**:
   - `MobileNetV2`: Ultra-low latency edge screening.
   - `ResNet-50`: Deep residual representation of irregular lesion boundaries.
   - `ViT-B/16`: Vision Transformer capturing long-range spatial context.
   - `Ensemble`: Soft-voting weighted consensus across CNNs and Vision Transformer.
3. **Lesion Segmentation (U-Net)**:
   - Evaluated on ISBI 2016 Part 1 dataset.
   - Generates binary segmentation mask and blended lesion contour overlay.
4. **Explainable AI (Grad-CAM)**:
   - Hooks final convolution gradients.
   - Visualizes positive class activation via JET thermal heatmaps.
   - Computes quantitative attention-lesion overlap metrics.

### 2.3 Automated Reporting Subsystem
- **PDF Report (`application/pdf`)**: Built via ReportLab 5.1.0 with institutional styling, patient metadata table, physician consultation block, prediction distribution charts, and medical liability disclaimers.
- **CSV Report (`text/csv`)**: Flat tabular export suitable for spreadsheet tools or batch electronic medical record (EMR) ingestion.
- **JSON Report (`application/json`)**: Complete hierarchical schema payload suitable for third-party integrations and FHIR adapters.

---

## 3. Production Deployment Topologies

### 3.1 Docker Compose Deployment (Recommended for On-Prem / VM)
Ideal for single-node Linux servers (e.g., Ubuntu 22.04 LTS on AWS EC2, GCP Compute Engine, or Azure VM):
1. Install Docker & Docker Compose plugin.
2. Clone repository:
   ```bash
   git clone <repo_url> /opt/dermai
   cd /opt/dermai
   ```
3. Configure environment:
   ```bash
   cp .env.docker .env
   # Update SECRET_KEY with: openssl rand -hex 32
   ```
4. Start system:
   ```bash
   docker compose up -d --build
   ```
5. Configure reverse proxy (Nginx / Caddy / Cloudflare Tunnel) with SSL/TLS termination pointing to port `5173`.

### 3.2 Cloud Container Services (AWS ECS / GCP Cloud Run)
- **Database**: Managed PostgreSQL (AWS RDS PostgreSQL or GCP Cloud SQL).
- **Backend Service**:
  - Build `backend/Dockerfile` and push to AWS ECR / GCP Artifact Registry.
  - Provision with 2 vCPUs and 4GB RAM minimum.
  - Set environment variables (`DATABASE_URL`, `SECRET_KEY`, `CORS_ORIGINS`).
  - Mount persistent EFS / Cloud Storage bucket for image uploads and model artifacts.
- **Frontend Service**:
  - Serve pre-built static bundle via AWS S3 + CloudFront CDN or GCP Cloud Storage + Cloud CDN.

---

## 4. Security & Compliance Architecture

1. **Authentication**: Stateless HMAC-SHA256 JWT tokens. Password hashing via Passlib / Bcrypt with 12 rounds.
2. **Audit Logging & Privacy**:
   - Sensitive fields (`password`, `access_token`, `authorization` headers) are explicitly sanitized from application logs.
   - Only diagnostic request paths, client IP, HTTP method, and response latency in ms are stored.
3. **CORS Boundary**: Configured strictly through `CORS_ORIGINS` environment variables.
4. **Input Sanitization**:
   - Real binary header sniffing via `python-magic` / Pillow verification to reject disguised malicious files.
   - Upload dimension limits and file size caps (max 15MB) to mitigate denial-of-service (DoS) memory exhaustion.
5. **Medical Device Disclaimer**:
   - All reports and UI views display prominent notices indicating DermAI is a decision-support research prototype and not an autonomous diagnostic instrument.
