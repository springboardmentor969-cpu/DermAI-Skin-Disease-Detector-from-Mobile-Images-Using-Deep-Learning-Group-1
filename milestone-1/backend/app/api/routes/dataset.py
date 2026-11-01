from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/dataset", tags=["dataset"])
PROJECT_ROOT = Path(__file__).resolve().parents[4]
CLASS_DISTRIBUTION_PATH = PROJECT_ROOT / "reports" / "class_distribution.csv"
METADATA_PATH = PROJECT_ROOT / "HAM10000_metadata.csv"


@lru_cache(maxsize=1)
def _read_dataset_statistics() -> dict[str, object]:
    if not CLASS_DISTRIBUTION_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError(
            "Dataset inspection artifacts are missing. Run scripts/inspect_dataset.py."
        )

    with CLASS_DISTRIBUTION_PATH.open(newline="", encoding="utf-8") as csv_file:
        class_counts = {
            row["dx"]: int(row["count"])
            for row in csv.DictReader(csv_file)
            if row.get("dx") and row.get("count")
        }
    if not class_counts:
        raise ValueError("The dataset class-distribution report contains no class counts.")

    image_count = 0
    lesion_ids: set[str] = set()
    with METADATA_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not {"image_id", "lesion_id", "dx"}.issubset(reader.fieldnames or set()):
            raise ValueError("The HAM10000 metadata is missing required columns.")
        for row in reader:
            if row.get("image_id"):
                image_count += 1
            if row.get("lesion_id"):
                lesion_ids.add(row["lesion_id"])

    if image_count != sum(class_counts.values()):
        raise ValueError(
            "The dataset report and metadata row counts differ. Regenerate the dataset report."
        )

    return {
        "dataset_name": "HAM10000",
        "image_type": "Dermoscopic skin-lesion images; not ordinary mobile-phone photographs.",
        "image_count": image_count,
        "unique_lesion_count": len(lesion_ids),
        "class_count": len(class_counts),
        "class_counts": class_counts,
        "source": "Existing HAM10000_metadata.csv and reports/class_distribution.csv",
    }


@router.get("/stats")
def get_dataset_statistics() -> dict[str, object]:
    try:
        return _read_dataset_statistics()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to read dataset statistics: {exc}",
        ) from exc
