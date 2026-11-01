from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.core.config import settings


class ImagePreprocessor:
    def __init__(self, target_size: tuple[int, int] | None = None):
        self.target_size = target_size or settings.TARGET_IMAGE_SIZE

    def validate(self, file_bytes: bytes) -> None:
        if len(file_bytes) <= 0:
            raise ValueError("Uploaded file is empty.")
        try:
            with Image.open(io.BytesIO(file_bytes)) as image:
                image.verify()
        except Exception as exc:  # pragma: no cover - defensive path
            raise ValueError("Corrupted or unreadable image file.") from exc

    def preprocess(self, file_bytes: bytes) -> tuple[np.ndarray, dict[str, Any]]:
        self.validate(file_bytes)
        with Image.open(io.BytesIO(file_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image = image.resize(self.target_size)
            array = np.asarray(image, dtype=np.float32)
            normalized = array / 255.0
            metadata = {
                "width": image.width,
                "height": image.height,
                "channels": image.mode,
                "source_format": "RGB",
                "normalization": settings.NORMALIZATION,
                "target_size": list(self.target_size),
                "array_shape": list(normalized.shape),
            }
            return normalized, metadata
