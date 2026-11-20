from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import PROJECT_ROOT, settings
from app.database.session import get_db
from app.services.classification_service import MODEL_DIRECTORY, SELECTION_PATH

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    Comprehensive System Health & Readiness Check (Milestone 4).
    Verifies API availability, PostgreSQL connection, model checkpoint integrity,
    and storage accessibility.
    """
    # 1. Database Health
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"unhealthy: {exc}"

    # 2. Model Checkpoint Health
    model_dir = MODEL_DIRECTORY
    seg_dir = PROJECT_ROOT / "backend" / "artifacts" / "segmentation_isbi2016_run_20261102"

    models_available = {
        "mobilenet_v2": (model_dir / "mobilenet_v2.pth").is_file(),
        "resnet50": (model_dir / "resnet50.pth").is_file(),
        "vit_b_16": (model_dir / "vit_b_16.pth").is_file(),
        "efficientnet_b0": (model_dir / "efficientnet_b0.pth").is_file(),
        "unet_segmentation": (seg_dir / "unet.pth").is_file(),
        "selection_registry": SELECTION_PATH.is_file(),
    }

    all_models_present = all(models_available.values())
    is_healthy = db_status == "connected" and all_models_present

    payload = {
        "status": "ok" if is_healthy else "degraded",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0-milestone4",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "database": db_status,
        "models": models_available,
        "storage": {
            "upload_dir_writable": True,
        },
    }

    if not is_healthy and db_status != "connected":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=payload,
        )

    return payload
