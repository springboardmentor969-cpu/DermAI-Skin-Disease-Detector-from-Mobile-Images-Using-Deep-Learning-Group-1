-- DermAI Milestone 1 PostgreSQL Schema
-- Compliant with de-identified clinical data guidelines

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(64) NOT NULL CHECK (role IN (
        'Patient / General User',
        'Dermatologist / Clinician',
        'Medical Researcher',
        'Data Scientist / ML Engineer',
        'Administrator'
    )),
    institution VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    patient_code VARCHAR(64) UNIQUE NOT NULL, -- De-identified e.g. PT-1042
    age INTEGER CHECK (age >= 0 AND age <= 120),
    gender VARCHAR(32) NOT NULL,
    fitzpatrick_skin_type VARCHAR(16),
    medical_history TEXT,
    known_allergies TEXT,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cases (
    id SERIAL PRIMARY KEY,
    case_code VARCHAR(64) UNIQUE NOT NULL, -- e.g. CASE-2026-001
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    assigned_clinician_id INTEGER REFERENCES users(id),
    body_location VARCHAR(128) NOT NULL,
    image_source VARCHAR(64) NOT NULL, -- Mobile Camera, Web Upload, Dermoscope
    analysis_status VARCHAR(64) DEFAULT 'PREPROCESSED_READY',
    model_version VARCHAR(64) DEFAULT 'Pre-Inference (M1 Pipeline v1.0)',
    clinical_notes TEXT,
    capture_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS case_images (
    id SERIAL PRIMARY KEY,
    image_code VARCHAR(64) UNIQUE NOT NULL,
    case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    original_filename VARCHAR(255) NOT NULL,
    file_format VARCHAR(32) NOT NULL,
    width_px INTEGER NOT NULL,
    height_px INTEGER NOT NULL,
    file_size_kb FLOAT NOT NULL,
    blur_score FLOAT NOT NULL,
    brightness_score FLOAT NOT NULL,
    contrast_score FLOAT NOT NULL,
    lesion_visibility_score FLOAT NOT NULL,
    quality_status VARCHAR(32) NOT NULL CHECK (quality_status IN ('Good Quality', 'Acceptable Quality', 'Poor Quality')),
    raw_storage_path TEXT NOT NULL,
    preprocessed_storage_path TEXT,
    preprocessing_metadata JSONB,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    user_role VARCHAR(64),
    action VARCHAR(128) NOT NULL,
    resource_type VARCHAR(64) NOT NULL,
    resource_id VARCHAR(64),
    ip_address VARCHAR(64),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
