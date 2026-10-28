import json
import uuid
from typing import List
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import User, Patient, Case, CaseImage, AuditLog
from .schemas import (
    UserCreate,
    UserLogin,
    UserResponse,
    TokenResponse,
    CaseCreate,
    PreprocessingResult,
    VALID_ROLES,
)
from .auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    record_audit_log,
)
from .services.preprocessing import run_preprocessing_pipeline

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="DermAI Milestone 1 API",
    description="AI-Powered Skin Disease Detector from Mobile Images — Milestone 1 Core Setup, Auth, Case Management & Preprocessing API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def seed_default_data():
    db = next(get_db())
    if db.query(User).count() == 0:
        demo_users = [
            User(
                email="clinician@dermai.org",
                full_name="Dr. Ananya Sharma (Dermatologist)",
                hashed_password=get_password_hash("DermAI@2026"),
                role="Dermatologist / Clinician",
                institution="AIIMS Dermatology Research Unit",
            ),
            User(
                email="patient@dermai.org",
                full_name="Rohan Verma (Patient)",
                hashed_password=get_password_hash("DermAI@2026"),
                role="Patient / General User",
                institution="Self-Registered Mobile User",
            ),
            User(
                email="ml@dermai.org",
                full_name="Vikram Nair (ML Engineer)",
                hashed_password=get_password_hash("DermAI@2026"),
                role="Data Scientist / ML Engineer",
                institution="DermAI Core AI Lab",
            ),
            User(
                email="researcher@dermai.org",
                full_name="Dr. Priya Menon (Medical Researcher)",
                hashed_password=get_password_hash("DermAI@2026"),
                role="Medical Researcher",
                institution="Clinical Dermatology Consortium",
            ),
            User(
                email="admin@dermai.org",
                full_name="System Administrator",
                hashed_password=get_password_hash("DermAI@2026"),
                role="Administrator",
                institution="DermAI Cloud Ops",
            ),
        ]
        db.add_all(demo_users)
        db.commit()

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "operational",
        "milestone": "Milestone 1 (Week 1 & 2) — Project Initialization, Dataset & Core Setup",
        "disclaimer": "Screening/research and clinical decision-support system only.",
    }

@app.post("/api/v1/auth/register", response_model=TokenResponse)
def register_user(payload: UserCreate, db: Session = Depends(get_db)):
    if payload.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role. Allowed: {VALID_ROLES}")
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=get_password_hash(payload.password),
        role=payload.role,
        institution=payload.institution,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    record_audit_log(db, user.email, user.role, "USER_REGISTER", "User", str(user.id))
    token = create_access_token({"sub": user.email, "role": user.role})
    return {"access_token": token, "token_type": "bearer", "user": user}

@app.post("/api/v1/auth/login", response_model=TokenResponse)
def login_user(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    record_audit_log(db, user.email, user.role, "USER_LOGIN", "Session", str(user.id))
    token = create_access_token({"sub": user.email, "role": user.role})
    return {"access_token": token, "token_type": "bearer", "user": user}

@app.get("/api/v1/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/api/v1/cases")
def create_case(payload: CaseCreate, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.patient_code == payload.patient_code).first()
    if not patient:
        patient = Patient(
            patient_code=payload.patient_code,
            age=payload.age,
            gender=payload.gender,
            medical_history=payload.medical_history,
            known_allergies=payload.known_allergies,
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

    case_code = f"CASE-{uuid.uuid4().hex[:6].upper()}"
    new_case = Case(
        case_code=case_code,
        patient_id=patient.id,
        body_location=payload.body_location,
        image_source=payload.image_source,
        clinical_notes=payload.clinical_notes,
    )
    db.add(new_case)
    db.commit()
    db.refresh(new_case)
    record_audit_log(db, "system@dermai.org", "Dermatologist / Clinician", "CREATE_CASE", "Case", case_code)
    return {
        "case_code": new_case.case_code,
        "patient_code": patient.patient_code,
        "age": patient.age,
        "gender": patient.gender,
        "body_location": new_case.body_location,
        "image_source": new_case.image_source,
        "analysis_status": new_case.analysis_status,
        "clinical_notes": new_case.clinical_notes,
        "capture_date": new_case.capture_date.isoformat(),
    }

@app.get("/api/v1/cases")
def list_cases(db: Session = Depends(get_db)):
    cases = db.query(Case).order_by(Case.capture_date.desc()).all()
    results = []
    for c in cases:
        results.append({
            "case_code": c.case_code,
            "patient_code": c.patient.patient_code if c.patient else "PT-UNKNOWN",
            "age": c.patient.age if c.patient else 0,
            "gender": c.patient.gender if c.patient else "Unknown",
            "body_location": c.body_location,
            "image_source": c.image_source,
            "analysis_status": c.analysis_status,
            "model_version": c.model_version,
            "clinical_notes": c.clinical_notes,
            "capture_date": c.capture_date.isoformat(),
            "image_count": len(c.images),
        })
    return results

@app.post("/api/v1/preprocess/upload", response_model=PreprocessingResult)
async def upload_and_preprocess(
    case_code: str = Form("CASE-DEMO-01"),
    crop_ratio: float = Form(0.90),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    content = await file.read()
    result = run_preprocessing_pipeline(content, filename=file.filename, target_size=224, crop_ratio=crop_ratio)
    image_code = f"IMG-{uuid.uuid4().hex[:6].upper()}"

    case_obj = db.query(Case).filter(Case.case_code == case_code).first()
    if case_obj:
        qm = result["quality_metrics"]
        img_record = CaseImage(
            image_code=image_code,
            case_id=case_obj.id,
            original_filename=file.filename,
            file_format=qm["format"],
            width_px=qm["width_px"],
            height_px=qm["height_px"],
            file_size_kb=qm["file_size_kb"],
            blur_score=qm["blur_laplacian_var"],
            brightness_score=qm["brightness_mean"],
            contrast_score=qm["contrast_std"],
            lesion_visibility_score=qm["lesion_visibility_score"],
            quality_status=qm["quality_status"],
            raw_storage_path=f"uploads/raw/{image_code}.png",
            preprocessed_storage_path=f"uploads/processed/{image_code}_norm.png",
            preprocessing_metadata_json=json.dumps(result["normalization_stats"]),
        )
        db.add(img_record)
        db.commit()

    record_audit_log(db, "clinician@dermai.org", "Dermatologist / Clinician", "IMAGE_PREPROCESS", "CaseImage", image_code)
    return {
        "image_code": image_code,
        "case_code": case_code,
        **result,
    }

@app.get("/api/v1/audit-logs")
def get_audit_logs(db: Session = Depends(get_db)):
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(50).all()
    return logs
