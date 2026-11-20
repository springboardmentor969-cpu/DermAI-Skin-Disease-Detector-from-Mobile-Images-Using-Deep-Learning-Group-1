from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(BASE_DIR / ".env", override=False)


class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "DermAI")
    API_V1_PREFIX: str = "/api/v1"
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/dermai",
    )
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    )
    DATASET_ROOT: Path = Path(os.getenv("DATASET_ROOT", PROJECT_ROOT))
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))
    MAX_IMAGE_DIMENSION: int = int(os.getenv("MAX_IMAGE_DIMENSION", "4096"))
    ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
    CORS_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]
    TARGET_IMAGE_SIZE = (224, 224)
    NORMALIZATION = "scale_0_1"


settings = Settings()
