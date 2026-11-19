# DermAI System Architecture

This document describes the high-level software architecture, component relationships, data flow, and interaction protocols across the DermAI screening platform.

---

## 1. System Architecture Diagram

```mermaid
graph TB
    subgraph ClientLayer ["Client Layer (Web Application)"]
        UI["React 19 + Vite + Tailwind CSS"]
        AuthCtx["Auth Context (JWT State)"]
        Crop["Image Cropping (react-image-crop)"]
        Axios["Axios API Client (Token Interceptors)"]
        
        UI --> AuthCtx
        UI --> Crop
        Crop --> Axios
        AuthCtx --> Axios
    end

    subgraph APILayer ["API Backend (FastAPI Service)"]
        Gateway["FastAPI Application (Port 8000)"]
        CORS["CORS Middleware"]
        AuthRouter["/auth Router (Login, Register)"]
        UserRouter["/users Router (Profile /me)"]
        CaseRouter["/cases Router (Cases & Uploads)"]
        HealthRouter["/health Diagnostics"]
        
        Deps["Route Dependencies (require_role, get_current_user)"]
        AuthService["Auth Service (Bcrypt, PyJWT, Audit Logging)"]
        StorageService["Storage Service (Boto3 MinIO Client)"]
        
        Gateway --> CORS
        CORS --> AuthRouter
        CORS --> UserRouter
        CORS --> CaseRouter
        CORS --> HealthRouter
        
        AuthRouter --> AuthService
        UserRouter --> Deps
        CaseRouter --> Deps
        CaseRouter --> StorageService
        CaseRouter --> AuthService
    end

    subgraph MLLayer ["Machine Learning Subsystem"]
        QualityMod["quality.py (Laplacian Blur, Brightness, Contrast)"]
        PreprocessMod["preprocess.py (EXIF, Crop, Resize, ImageNet Norm)"]
        PyTorchEngine["PyTorch Engine (Shape: 3x224x224, float32)"]
        
        CaseRouter --> QualityMod
        QualityMod --> PreprocessMod
        PreprocessMod --> PyTorchEngine
    end

    subgraph DataLayer ["Data & Storage Layer (Docker Compose)"]
        Postgres[("PostgreSQL 16 Database\n(Port 5432)\n• users\n• patients\n• cases\n• images\n• predictions\n• audit_logs")]
        MinIO[("MinIO Object Storage\n(S3 API Port 9000, Console Port 9001)\nBucket: dermai-images")]
        
        CaseRouter -->|SQLAlchemy ORM / Sessions| Postgres
        AuthService -->|Persist Users & Audit Logs| Postgres
        StorageService -->|Boto3 S3 API / Stream Bytes| MinIO
    end

    %% Network communication
    Axios -->|HTTP REST / JSON / Multipart (Port 8000)| Gateway
```

---

## 2. Component Descriptions

### 2.1 Frontend Web Application (`frontend/`)
- **Technology Stack**: React 19, Vite, Tailwind CSS v4, Lucide React icons, and `react-image-crop`.
- **Key Responsibilities**:
  - Provides responsive, calm medical interfaces for patient case creation, camera capture, and clinician review.
  - Implements client-side framing and interactive 1:1 square cropping to isolate lesions.
  - Manages JWT tokens in `localStorage` with automated session recovery and logout on 401 Unauthorized errors.
  - Displays color-coded automated quality feedback badges (`good`, `acceptable`, `poor`).

### 2.2 Backend API Gateway (`backend/app/`)
- **Technology Stack**: FastAPI, Pydantic v2 Settings, SQLAlchemy 2.0, Alembic, Passlib/Bcrypt, PyJWT, and Boto3.
- **Key Responsibilities**:
  - **Authentication & RBAC**: Provides secure password hashing via bcrypt, signed Bearer JWTs, and the reusable [`require_role(...)`](file:///c:/Users/2005s/Downloads/DermAI/backend/app/dependencies.py) dependency.
  - **Case Management**: Enforces role isolation (patients access only their submitted cases, clinicians access all cases).
  - **Upload Processing**: Coordinates binary validation, quality assessment, MinIO persistence, and database transactions.
  - **Audit Logging**: Persists tamper-evident records for every login attempt, case creation, and image upload.

### 2.3 Machine Learning & Preprocessing (`ml/preprocessing/`)
- **Technology Stack**: OpenCV (`opencv-python-headless`), Pillow, NumPy.
- **Key Modules**:
  1. [`quality.py`](file:///c:/Users/2005s/Downloads/DermAI/ml/preprocessing/quality.py): Evaluates blur (Laplacian variance), illumination (mean intensity), and RMS contrast against centralized clinical thresholds in `QUALITY_THRESHOLDS`.
  2. [`preprocess.py`](file:///c:/Users/2005s/Downloads/DermAI/ml/preprocessing/preprocess.py): Standardizes clinical images through EXIF orientation correction, center-cropping, 224×224 bicubic resizing, and ImageNet channel normalization, returning contiguous PyTorch-ready tensors.

### 2.4 Persistence & Object Storage Layer (`docker-compose.yml`)
- **PostgreSQL 16**: Relational storage for structured records, foreign-key relationships, and indexing. Data is persisted to the named volume `dermai_postgres_data`.
- **MinIO Object Storage**: High-performance, S3-compatible storage for binary photograph files (`.jpg`, `.png`). Data is persisted to the named volume `dermai_minio_data`. Buckets (`dermai-images`) are automatically verified and created on startup via Boto3.

---

## 3. Communication Protocols & Security

```
Client (Browser) ──[ HTTPS / Bearer JWT ]──> FastAPI (8000)
FastAPI (8000)   ──[ TCP / PostgreSQL Wire ]──> PostgreSQL (5432)
FastAPI (8000)   ──[ S3 Signature v4 / HTTP ]──> MinIO (9000)
```

1. **Authentication Flow**:
   - Clients send credentials to `POST /auth/login`.
   - FastAPI verifies bcrypt password hash and returns a signed HS256 JWT access token with expiry (`exp`) and subject claim (`sub: email`).
   - Clients attach `Authorization: Bearer <token>` to subsequent requests.
2. **CORS Policy**: Configured in FastAPI to allow cross-origin requests from the React development server while preserving credentials.
3. **Storage Security**: Photographs in MinIO are stored under non-enumerable UUID keys (`cases/{case_id}/{uuid}.jpg`).
