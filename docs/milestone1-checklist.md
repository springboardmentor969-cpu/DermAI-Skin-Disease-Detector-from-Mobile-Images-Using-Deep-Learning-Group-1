# DermAI Milestone 1: Evaluation Checklist

This checklist audits the deliverables and technical evaluation criteria for **Milestone 1** of the DermAI platform.

---

## 1. Architecture & Infrastructure
- [x] **Docker Compose Configuration**: `docker-compose.yml` configures PostgreSQL 16 and MinIO object storage.
- [x] **MinIO Compatibility**: Updated to current compatible MinIO/S3 image (`pgsty/silo:RELEASE.2026-09-16T00-00-00Z`) resolving Docker Hub pull restrictions.
- [x] **Persistent Storage Volumes**: Named Docker volumes (`dermai_postgres_data`, `dermai_minio_data`) configured to guarantee zero data loss across container restarts.
- [x] **Automated Bucket Provisioning**: Boto3 client automatically verifies and creates the `dermai-images` bucket on demand.
- [x] **Service Healthchecks**: Live readiness checks configured for PostgreSQL (`pg_isready`) and MinIO (`mc ready local`).

---

## 2. Backend API Architecture (FastAPI & SQLAlchemy)
- [x] **Modular Project Structure**:
  - `app/main.py`: Entrypoint with CORS middleware, router registration, and root status.
  - `app/config.py`: Centralized Pydantic Settings reading environment variables from `.env`.
  - `app/database.py`: SQLAlchemy engine, session maker (`SessionLocal`), and `get_db` route dependency.
  - `app/models/`: Declarative models inheriting from `Base`.
  - `app/schemas/`: Pydantic validation and response schemas.
  - `app/routers/`: Modular route controllers (`health`, `auth`, `users`, `cases`).
  - `app/services/`: Service layer separating auth, cryptographic, and storage logic.
- [x] **Database Schema & Tables**:
  - [x] `users`: `id`, `email` (unique), `password_hash`, `full_name`, `role`, `created_at`.
  - [x] `patients`: `id`, `age`, `gender`, `notes`, `created_at` (strictly non-identifying demographics).
  - [x] `cases`: `id`, `patient_id` (FK), `created_by` (FK), `body_location`, `status`, `created_at`.
  - [x] `images`: `id`, `case_id` (FK), `storage_key`, `source`, `captured_at`, `quality_label`, `quality_details`.
  - [x] `predictions`: `id`, `image_id` (FK), `model_version`, `predicted_class`, `confidence`, `top_n`, `created_at`.
  - [x] `audit_logs`: `id`, `user_id` (FK), `action`, `details`, `created_at`.
- [x] **Database Migrations (Alembic)**:
  - [x] `alembic.ini` and `alembic/env.py` configured to connect to PostgreSQL from `.env`.
  - [x] Initial migration `94502164cedd_create_initial_tables.py` generated and applied (`alembic upgrade head`).
- [x] **Health Check Diagnostic**: `GET /health` endpoint verifying API operational status and active database connectivity.

---

## 3. Authentication & Role-Based Access Control
- [x] **Password Hashing**: Salted password hashing via Bcrypt using Passlib CryptContext (with safe 72-byte truncation compatibility for modern bcrypt).
- [x] **JWT Generation & Expiry**: Signed Bearer JWT tokens with configurable expiration using `SECRET_KEY`, `ALGORITHM`, and `ACCESS_TOKEN_EXPIRE_MINUTES`.
- [x] **User Registration (`POST /auth/register`)**:
  - [x] Public registration as `admin` is blocked (`403 Forbidden`).
  - [x] Duplicate emails are rejected (`400 Bad Request`).
  - [x] Successful and blocked registrations are logged to `audit_logs`.
- [x] **User Authentication (`POST /auth/login`)**:
  - [x] Accepts JSON payload or form-data (compatible with Swagger UI *Authorize* modal).
  - [x] Successful logins return access token and write `USER_LOGIN_SUCCESS` audit log.
  - [x] Failed password/user attempts return `401 Unauthorized` and log `USER_LOGIN_FAILED`.
- [x] **Current User Endpoint (`GET /users/me`)**: Returns profile for authenticated token holder.
- [x] **Reusable Role Dependency**: `require_role(*allowed_roles)` protects endpoints by user role (e.g., clinician, admin).

---

## 4. Case Management & Image Upload
- [x] **Create Case (`POST /cases`)**:
  - [x] Accepts patient demographics (`age`, `gender`, `notes`) and anatomical `body_location`.
  - [x] Automatically provisions de-identified `Patient` record.
  - [x] Logs `CASE_CREATED` in `audit_logs`.
- [x] **List Cases (`GET /cases`)**:
  - [x] Role-based isolation: Patients view only their own cases.
  - [x] Clinicians, researchers, and administrators view all system cases.
- [x] **Retrieve Case (`GET /cases/{id}`)**:
  - [x] Ownership enforced: Patients attempting to view another patient's case receive `403 Forbidden`.
- [x] **Image Upload (`POST /cases/{id}/images`)**:
  - [x] Validates format: Rejects non-JPG/PNG files.
  - [x] Validates size: Rejects files larger than 10 MB.
  - [x] Validates resolution: Rejects images below 224 × 224 pixels.
  - [x] Pillow Content Verification: Verifies genuine image binary structure, not just extension.
  - [x] Boto3 MinIO Persistence: Saves image under unique UUID key `cases/{case_id}/{uuid}.{jpg|png}`.
  - [x] Quality Assessment: Automatically runs `assess_image_quality`, storing `quality_label` and `quality_details`.
  - [x] Poor Quality Retake Warning: Preserves poor images but returns warning advising retake in better lighting.
  - [x] Audit Logging: Logs `IMAGE_UPLOADED` with file size, dimensions, and quality metrics.
- [x] **List Images (`GET /cases/{id}/images`)**: Returns photograph history and quality details for the case.

---

## 5. Machine Learning & Preprocessing Pipeline
- [x] **Image Quality Assessment (`ml/preprocessing/quality.py`)**:
  - [x] Centralized, configurable threshold dictionary: `QUALITY_THRESHOLDS` at top of module.
  - [x] Blur score via Laplacian variance (`cv2.Laplacian(gray, cv2.CV_64F).var()`).
  - [x] Illumination / brightness assessment (mean pixel intensity, 0-255).
  - [x] Contrast assessment (RMS pixel intensity standard deviation).
  - [x] Categorizes into `good`, `acceptable`, or `poor` with explanation list (`issues`).
- [x] **Preprocessing Pipeline (`ml/preprocessing/preprocess.py`)**:
  - [x] Step 1: Image validation and integrity checks.
  - [x] Step 2: EXIF orientation correction (`ImageOps.exif_transpose`).
  - [x] Step 3: Symmetrical center-crop to square (1:1).
  - [x] Step 4: Bicubic resizing to 224 × 224.
  - [x] Step 5: Conversion to 3-channel RGB.
  - [x] Step 6: ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
  - [x] Step 7: Formatted as PyTorch-ready array (shape: `(3, 224, 224)`, dtype: `float32`, channels-first).
- [x] **Preprocessing Benchmark Script (`test_preprocessing.py`)**:
  - [x] Synthesizes diverse test conditions (sharp, blurry, dark, washed out, low contrast).
  - [x] Prints formatted ASCII comparison table with metrics and tensor shapes.
- [x] **Exploratory Data Analysis Notebook (`ml/notebooks/01_eda.ipynb`)**:
  - [x] Full analysis of the 10,015 HAM10000 images and metadata.
  - [x] Detailed diagnosis mapping (`nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`).
  - [x] Missing value audit (57 age nulls) and zero duplicate image IDs verified.
  - [x] Annotated class frequency bar charts demonstrating class imbalance.
  - [x] Demographic analysis (age, sex, anatomical localizations, cross-tab heatmaps).
  - [x] Visual grid of 3 sample dermoscopy images per class (21 total).
  - [x] Physical image resolution verification (600 × 450 RGB).
  - [x] Data Leakage explanation: why splitting by `lesion_id` is mandatory to avoid skin artifact memorization.
  - [x] Grouped & stratified 70/15/15 train/val/test split generated and exported to `ml/data/splits.csv`.
  - [x] Clinical discussion of the dermoscopic vs smartphone photo domain gap.

---

## 6. Frontend Web Application (React + Vite + Tailwind CSS)
- [x] **Design System & Aesthetics**: Calm medical styling using slate and sky blues, rounded cards, generous spacing, and Inter typography.
- [x] **Mandatory Medical Disclaimer**: Prominent footer note on every page: *"AI-assisted screening tool. Not a medical diagnosis."*
- [x] **Authentication Pages**:
  - [x] `LoginPage`: Email/password form, validation, and error alerts.
  - [x] `RegisterPage`: Full registration with role selector (patient, clinician, researcher, ML engineer).
  - [x] Safe JWT storage in `localStorage` with auto-session restoration.
  - [x] `ProtectedRoute`: Route guard redirecting unauthenticated traffic to `/login`.
- [x] **Dashboard (`DashboardPage`)**:
  - [x] Role-aware welcome banner.
  - [x] Case metrics summary cards (Cases, Scans, Status).
  - [x] Primary "New Case" trigger button.
  - [x] Recent cases list with body location, dates, patient demographics, and scan counts.
- [x] **New Case Form (`NewCasePage`)**:
  - [x] Patient age and gender inputs.
  - [x] Body location field with clickable anatomical site chips.
  - [x] Clinical observations textarea.
  - [x] Auto-redirects to image upload upon creation.
- [x] **Image Upload & Cropping (`UploadPage`)**:
  - [x] Option 1: Mobile camera capture (`capture="environment"`).
  - [x] Option 2: Gallery file picker (`.jpg`, `.png`).
  - [x] Interactive 1:1 square cropping using `react-image-crop`.
  - [x] Post-upload quality assessment card displaying blur, brightness, contrast, and resolution.
  - [x] Prominent warning banner if image quality is poor advising retake.
- [x] **Case Detail View (`CaseDetailPage`)**:
  - [x] Case overview card with demographics and body location.
  - [x] Photograph gallery with color-coded `QualityBadge` components.
  - [x] MinIO storage key references and quality details.

---

## 7. Automated Testing & Verification
- [x] **Backend Pytest Suite (`backend/tests/`)**:
  - [x] 16 passing automated integration tests.
  - [x] Isolated SQLite in-memory test database fixture ensuring zero pollution of production database.
  - [x] Coverage: Registration success, admin restriction, duplicate emails, login success, wrong password, protected routes, `require_role` dependency, case creation, patient vs clinician visibility, valid upload, poor quality retake warning, wrong file type, oversized file (>10MB), low resolution (<224x224), and binary image content streaming.
- [x] **Frontend Production Build Verification**: `npm run build` generates zero errors and bundles optimized production assets.
