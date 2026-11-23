"""
Application configuration management using Pydantic Settings.
Reads settings from the root .env file and environment variables.
"""

import sys
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Locate the root .env file (one or two levels up from this config file)
BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ENV_FILE = ROOT_DIR / ".env" if (ROOT_DIR / ".env").exists() else BASE_DIR / ".env"


class Settings(BaseSettings):
    # CORS and Host Security
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    ALLOWED_HOSTS: str = "localhost,127.0.0.1"

    # Application Info
    PROJECT_NAME: str = "DermAI Screening & Decision Support API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Database Configuration
    DATABASE_URL: str = "postgresql://dermai_user:dermai_password@localhost:5432/dermai_db"
    POSTGRES_USER: str = "dermai_user"
    POSTGRES_PASSWORD: str = "dermai_password"
    POSTGRES_DB: str = "dermai_db"
    POSTGRES_PORT: int = 5432

    # MinIO (S3-Compatible Object Storage) Configuration
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ROOT_USER: str = "minioadmin"
    MINIO_ROOT_PASSWORD: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "dermai-images"
    MINIO_USE_SSL: bool = False

    # Security & JWT Authentication
    SECRET_KEY: str = "supersecretkey_change_in_production_1234567890abcdef"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # AI Model & Inference Configuration
    MODEL_PATH: str = "ml/experiments/resnet50_smoke/best.pt"
    MODEL_VERSION: str = "smoke-test"
    DEMO_MODEL: bool = True
    CONFIDENCE_THRESHOLD: float = 0.60
    ENABLE_SEGMENTATION: bool = True
    SEGMENTATION_MODEL_PATH: str = "ml/experiments/segmentation_smoke/best_model.pth"
    GRADCAM_ENABLED: bool = True
    RISK_SCORING_ENABLED: bool = True
    ENSEMBLE_MODELS: str = ""

    # Redis Cache Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_URL: Optional[str] = None
    REDIS_CACHE_TTL: int = 3600  # 1 hour in seconds

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE) if ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
