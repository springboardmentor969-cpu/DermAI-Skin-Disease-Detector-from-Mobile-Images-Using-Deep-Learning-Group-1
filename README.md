# DermAI — Clinical Skin Lesion Analysis & Decision Support Platform

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)
![React](https://img.shields.io/badge/React-18-blue.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-5.5-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![Tests](https://img.shields.io/badge/Tests-41%20Passed-brightgreen.svg)

DermAI is an end-to-end dermatological AI decision-support platform designed to analyze skin lesion photographs, assist clinicians with differential diagnoses, segment lesion boundaries, generate visual explainability heatmaps, and produce publication-ready clinical reports.

---

## 🌟 Key Features

- **Multi-Model Deep Learning**: Supports MobileNetV2, EfficientNet-B0, ResNet-50, Vision Transformer (ViT-B/16), and Soft-Voting Ensemble architectures trained on the HAM10000 dataset (10,015 dermoscopic images).
- **Explainable AI (Grad-CAM & Saliency)**: Visualizes model attention heatmaps and calculates lesion-attention overlap scores to prevent "black-box" clinical decisions.
- **Lesion Segmentation (U-Net)**: Automatically segments lesion contours and computes lesion area percentages using models trained on the ISBI 2016 dataset.
- **Role-Based Access Control (RBAC)**: Distinct workflows for **Patients** (case creation, scan uploads, personal reports) and **Doctors** (system-wide triage, clinical reviews, PDF exports).
- **Automated Clinical Reporting**: One-click generation of comprehensive clinical diagnostic reports in **PDF**, **CSV**, and **JSON** formats.
- **System Monitoring & Logging**: Structured application audit logs with credential masking and detailed multi-component `/health` monitoring.
- **Dockerized Multi-Container Stack**: Production-ready `docker-compose.yml` coordinating PostgreSQL 16, FastAPI backend, and Nginx reverse proxy frontend.

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- **Python 3.11+**
- **Node.js 20+**
- **PostgreSQL 14+** (running locally on port 5432)

### 2. Backend Setup
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env

# Start FastAPI backend (port 8000)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

### 3. Frontend Setup
```powershell
# In a separate terminal:
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🐳 Quick Start (Docker Compose)

Deploy the entire full-stack application with a single command:

```powershell
# 1. Prepare environment
Copy-Item .env.docker .env

# 2. Build and launch services
docker compose up -d --build

# 3. View status and logs
docker compose ps
docker compose logs -f
```

- **Web Application**: [http://localhost:5173](http://localhost:5173)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

Detailed Docker documentation: [docs/docker_setup.md](docs/docker_setup.md).

---

## 🔑 Test User Credentials

The landing page features quick one-click login buttons for testing:

| Role | Name | Email | Password | Access Capabilities |
|---|---|---|---|---|
| 👨‍⚕️ **Doctor** | **Dr. Arjun** | `doctor@dermai.com` | `doctor123` | System-wide cases, consultation review, PDF/CSV/JSON report export |
| 👤 **Patient** | **Varun** | `patient@dermai.com` | `patient123` | Personal case creation, image uploads, AI analysis view |

---

## 📊 Model Benchmark Results

| Model Architecture | Parameters | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | AUC-ROC | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mobilenet_v2` | 3.5M | 77.2% | 0.595 | 77.0% | 0.565 | 0.912 | 14 ms |
| `efficientnet_b0` | 5.3M | 75.1% | 0.572 | 74.8% | 0.549 | 0.908 | 23 ms |
| `resnet50` | 25.6M | 81.4% | 0.658 | 81.0% | 0.642 | 0.938 | 39 ms |
| `vit_b_16` | 86.6M | 82.9% | 0.684 | 82.4% | 0.671 | 0.949 | 72 ms |
| **`ensemble`** | **115.7M** | **84.6%** | **0.712** | **84.1%** | **0.698** | **0.962** | **95 ms** |

---

## 🧪 Comprehensive Testing Suite

DermAI includes 41 automated tests spanning unit, integration, RBAC, ML inference, and report exports:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest tests -v
```

```
================= 41 passed, 88 warnings in 149.42s =================
```

---

## 📚 Complete Project Documentation

- [System Architecture & Cloud Deployment](docs/architecture_and_deployment.md)
- [Docker Setup & Production Guide](docs/docker_setup.md)
- [RESTful API Specification](docs/api.md)
- [Database Schema & Data Dictionary](docs/database.md)
- [Deep Learning Models & Evaluation](docs/models_and_evaluation.md)
- [Final Milestone 4 Completion Report](reports/milestone4_final_report.md)

---

## ⚠️ Medical Device Disclaimer

DermAI is an artificial intelligence research and clinical decision-support prototype. It is **not** an FDA-approved medical diagnostic device and does not replace the professional clinical judgment of a licensed dermatologist.
