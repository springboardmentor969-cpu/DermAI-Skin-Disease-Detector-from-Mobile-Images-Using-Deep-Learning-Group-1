# DermAI — RESTful API Specification (v1)

Base URL: `/api/v1` (with root health at `/health`)  
Interactive OpenAPI UI: `http://localhost:8000/docs`  
ReDoc UI: `http://localhost:8000/redoc`

---

## 1. Authentication Endpoints

### 1.1 Register User / Doctor
- **Method**: `POST`
- **Path**: `/api/v1/auth/register`
- **Auth**: None
- **Request Body**:
  ```json
  {
    "email": "user@example.com",
    "password": "SecurePassword123!",
    "full_name": "Full Name",
    "role": "user"  // "user" (patient) or "doctor"
  }
  ```
- **Response** (`201 Created`):
  ```json
  {
    "id": 1,
    "email": "user@example.com",
    "full_name": "Full Name",
    "role": "user",
    "is_active": true
  }
  ```

### 1.2 Sign In / Login
- **Method**: `POST`
- **Path**: `/api/v1/auth/login`
- **Auth**: None
- **Request Body** (`application/x-www-form-urlencoded` or JSON):
  ```json
  {
    "username": "user@example.com",
    "password": "SecurePassword123!"
  }
  ```
- **Response** (`200 OK`):
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1Ni...",
    "token_type": "bearer",
    "user": {
      "id": 1,
      "email": "user@example.com",
      "full_name": "Full Name",
      "role": "user"
    }
  }
  ```

### 1.3 Get Current User Profile
- **Method**: `GET`
- **Path**: `/api/v1/auth/me`
- **Auth**: Bearer Token
- **Response** (`200 OK`): Current user object.

---

## 2. Cases & RBAC Management

### 2.1 List Cases
- **Method**: `GET`
- **Path**: `/api/v1/cases`
- **Auth**: Bearer Token
- **Behavior**:
  - If authenticated as `user`: Returns only cases owned by this user.
  - If authenticated as `doctor`: Returns all patient cases across the system.
- **Response** (`200 OK`): Array of Case objects.

### 2.2 Create Case
- **Method**: `POST`
- **Path**: `/api/v1/cases`
- **Auth**: Bearer Token
- **Request Body**:
  ```json
  {
    "title": "Suspected lesion on right forearm",
    "description": "Noticed color darkening over past 2 months"
  }
  ```
- **Response** (`201 Created`): Created case object.

### 2.3 Get Case Detail
- **Method**: `GET`
- **Path**: `/api/v1/cases/{case_id}`
- **Auth**: Bearer Token
- **Permissions**: Case owner or Doctor. Other patients receive `403 Forbidden`.
- **Response** (`200 OK`): Case object including associated images and predictions.

### 2.4 Doctor Clinical Review
- **Method**: `POST`
- **Path**: `/api/v1/cases/{case_id}/review`
- **Auth**: Bearer Token (`doctor` role only)
- **Request Body**:
  ```json
  {
    "review_status": "reviewed",
    "doctor_review_notes": "Clinical review completed. Advise biopsy."
  }
  ```
- **Response** (`200 OK`): Updated case record.

---

## 3. Image Upload & Preprocessing

### 3.1 Upload Dermoscopic Image
- **Method**: `POST`
- **Path**: `/api/v1/upload/{case_id}`
- **Auth**: Bearer Token (Case owner or Doctor)
- **Form Data**: `file` (JPEG or PNG, max 15MB)
- **Validation**: Strict MIME sniffing and Pillow header verification.
- **Response** (`201 Created`):
  ```json
  {
    "id": 12,
    "filename": "lesion_01.jpg",
    "width": 600,
    "height": 450,
    "color_mode": "RGB",
    "status": "uploaded",
    "quality": {
      "overall_quality": "good",
      "blur_score": 245.8,
      "exposure_score": 118.2
    }
  }
  ```

---

## 4. Deep Learning & Explainability (XAI)

### 4.1 List Available Models & Benchmarks
- **Method**: `GET`
- **Path**: `/api/v1/analysis/models`
- **Auth**: None
- **Response** (`200 OK`): Model benchmark registry (MobileNetV2, ResNet-50, ViT-B/16, EfficientNet-B0, Ensemble).

### 4.2 Analyze Image (Multi-Model & Grad-CAM)
- **Method**: `POST`
- **Path**: `/api/v1/analysis/images/{image_id}`
- **Auth**: Bearer Token
- **Query Params**: `model_name` (optional: `ensemble`, `mobilenet_v2`, `resnet50`, `vit_b_16`, `efficientnet_b0`)
- **Response** (`200 OK`):
  ```json
  {
    "image_id": 12,
    "model_name": "ensemble",
    "predicted_class": "mel",
    "confidence": 0.884,
    "entropy": 0.382,
    "risk_level": "High Risk",
    "top_predictions": [
      {"class": "mel", "label": "Melanoma", "probability": 0.884},
      {"class": "nv", "label": "Melanocytic Nevus", "probability": 0.082}
    ],
    "segmentation": {
      "mask_data_url": "data:image/png;base64,...",
      "overlay_data_url": "data:image/png;base64,...",
      "lesion_area_pct": 18.4
    },
    "gradcam": {
      "heatmap_data_url": "data:image/png;base64,...",
      "overlay_data_url": "data:image/png;base64,...",
      "attention_overlap_pct": 89.2
    }
  }
  ```

---

## 5. Automated Clinical Reporting Subsystem

### 5.1 Export Case Report
- **Method**: `GET`
- **Path**: `/api/v1/reports/cases/{case_id}`
- **Auth**: Bearer Token (Case owner or Doctor)
- **Query Params**: `format` (`pdf` | `csv` | `json`), default is `pdf`
- **Response Headers**:
  - `Content-Disposition: attachment; filename="DermAI_Report_Case_{case_id}.{format}"`
  - `Content-Type`:
    - `pdf`: `application/pdf`
    - `csv`: `text/csv; charset=utf-8`
    - `json`: `application/json`

### 5.2 Export Single Image Analysis Report
- **Method**: `GET`
- **Path**: `/api/v1/reports/images/{image_id}`
- **Auth**: Bearer Token
- **Query Params**: `format` (`pdf` | `csv` | `json`)
- **Response**: Binary or text attachment.

---

## 6. System Health & Dataset

### 6.1 System Health
- **Method**: `GET`
- **Path**: `/health` and `/api/v1/health`
- **Auth**: None
- **Response** (`200 OK`):
  ```json
  {
    "status": "ok",
    "service": "DermAI",
    "version": "1.0.0-milestone4",
    "timestamp": "2026-11-19T13:21:42.644896Z",
    "database": "connected",
    "models": {
      "mobilenet_v2": true,
      "resnet50": true,
      "vit_b_16": true,
      "efficientnet_b0": true,
      "unet_segmentation": true,
      "selection_registry": true
    },
    "storage": {
      "upload_dir_writable": true
    }
  }
  ```

### 6.2 Dataset Statistics
- **Method**: `GET`
- **Path**: `/api/v1/dataset/stats`
- **Auth**: None
- **Response** (`200 OK`): HAM10000 counts, class distribution, lesion totals, and validation splits.
