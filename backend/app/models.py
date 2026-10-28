from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(64), nullable=False, default="Patient / General User")
    institution = Column(String(255), default="DermAI Clinical Network")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    patient_code = Column(String(64), unique=True, index=True, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(32), nullable=False)
    fitzpatrick_skin_type = Column(String(16), default="Type III")
    medical_history = Column(Text, default="None reported")
    known_allergies = Column(Text, default="NKDA")
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    cases = relationship("Case", back_populates="patient", cascade="all, delete-orphan")

class Case(Base):
    __tablename__ = "cases"
    id = Column(Integer, primary_key=True, index=True)
    case_code = Column(String(64), unique=True, index=True, nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    body_location = Column(String(128), nullable=False)
    image_source = Column(String(64), nullable=False, default="Mobile Camera")
    analysis_status = Column(String(64), default="PREPROCESSED_READY")
    model_version = Column(String(64), default="M1-Preproc-v1.0")
    clinical_notes = Column(Text, default="")
    capture_date = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="cases")
    images = relationship("CaseImage", back_populates="case", cascade="all, delete-orphan")

class CaseImage(Base):
    __tablename__ = "case_images"
    id = Column(Integer, primary_key=True, index=True)
    image_code = Column(String(64), unique=True, index=True, nullable=False)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_format = Column(String(32), nullable=False)
    width_px = Column(Integer, nullable=False)
    height_px = Column(Integer, nullable=False)
    file_size_kb = Column(Float, nullable=False)
    blur_score = Column(Float, nullable=False)
    brightness_score = Column(Float, nullable=False)
    contrast_score = Column(Float, nullable=False)
    lesion_visibility_score = Column(Float, nullable=False)
    quality_status = Column(String(32), nullable=False)
    raw_storage_path = Column(Text, nullable=False)
    preprocessed_storage_path = Column(Text, nullable=True)
    preprocessing_metadata_json = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    case = relationship("Case", back_populates="images")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String(255))
    user_role = Column(String(64))
    action = Column(String(128), nullable=False)
    resource_type = Column(String(64), nullable=False)
    resource_id = Column(String(64))
    timestamp = Column(DateTime, default=datetime.utcnow)
