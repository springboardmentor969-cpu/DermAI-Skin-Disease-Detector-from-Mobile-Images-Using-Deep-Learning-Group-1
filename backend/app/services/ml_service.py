"""
ML Prediction Service with Redis Caching for DermAI.

Features:
- Cache key: sha256 hash of the input image bytes
- Cache TTL: 1 hour (3600 seconds)
- Instant cache hits on identical re-uploaded or re-analyzed images
- Resilient fallback to un-cached inference if Redis is offline or unavailable
"""

import io
import json
import logging
import hashlib
from typing import Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image as PILImage
import numpy as np

from app.config import settings
from app.services.inference import inference_service
from ml.ensemble.ensemble_predict import ensemble_predict

logger = logging.getLogger("dermai.ml_service")

# Default TTL: 1 hour (3600 seconds)
CACHE_TTL_SECONDS = getattr(settings, "REDIS_CACHE_TTL", 3600)


class MLService:
    """
    ML Prediction Service with Redis caching layer.
    """

    def __init__(self):
        self._redis = None
        self._redis_checked = False
        self._redis_available = False

    def _get_redis(self):
        """Lazy initialization of Redis client with non-blocking error handling."""
        if self._redis is not None:
            return self._redis

        try:
            import redis
            redis_url = getattr(settings, "REDIS_URL", None)
            if redis_url:
                client = redis.Redis.from_url(
                    redis_url,
                    socket_connect_timeout=1.0,
                    socket_timeout=1.0,
                    decode_responses=True,
                )
            else:
                client = redis.Redis(
                    host=getattr(settings, "REDIS_HOST", "localhost"),
                    port=getattr(settings, "REDIS_PORT", 6379),
                    db=getattr(settings, "REDIS_DB", 0),
                    password=getattr(settings, "REDIS_PASSWORD", None),
                    socket_connect_timeout=1.0,
                    socket_timeout=1.0,
                    decode_responses=True,
                )

            # Test connection
            client.ping()
            self._redis = client
            self._redis_available = True
            logger.info("[MLService] Successfully connected to Redis prediction cache.")
        except Exception as e:
            self._redis = None
            self._redis_available = False
            logger.debug(f"[MLService] Redis cache not available ({e}). Proceeding without cache.")

        return self._redis

    @property
    def is_cache_available(self) -> bool:
        """Returns True if Redis is reachable."""
        r = self._get_redis()
        return r is not None and self._redis_available

    def compute_image_hash(
        self,
        image_input: Union[str, Path, bytes, bytearray, PILImage.Image, np.ndarray],
    ) -> str:
        """
        Computes SHA-256 hash of image bytes for cache key.
        """
        if isinstance(image_input, (bytes, bytearray)):
            raw_bytes = bytes(image_input)
        elif isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if p.is_file():
                raw_bytes = p.read_bytes()
            else:
                raw_bytes = str(image_input).encode("utf-8")
        elif isinstance(image_input, PILImage.Image):
            buf = io.BytesIO()
            fmt = image_input.format or "PNG"
            image_input.save(buf, format=fmt)
            raw_bytes = buf.getvalue()
        elif isinstance(image_input, np.ndarray):
            raw_bytes = image_input.tobytes()
        else:
            raw_bytes = str(image_input).encode("utf-8")

        return hashlib.sha256(raw_bytes).hexdigest()

    def get_cached_prediction(
        self,
        image_input: Union[str, Path, bytes, bytearray, PILImage.Image, np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves prediction from Redis cache by sha256 hash of image bytes.
        Returns None if cache miss or Redis is unavailable.
        """
        r = self._get_redis()
        if r is None:
            return None

        try:
            img_hash = self.compute_image_hash(image_input)
            cache_key = f"dermai:pred:{img_hash}"
            cached_val = r.get(cache_key)
            if cached_val:
                logger.info(f"[MLService] Cache HIT for image hash {img_hash[:12]}...")
                data = json.loads(cached_val)
                data["cached"] = True
                return data
        except Exception as e:
            logger.debug(f"[MLService] Redis read failed: {e}")

        return None

    def set_cached_prediction(
        self,
        image_input: Union[str, Path, bytes, bytearray, PILImage.Image, np.ndarray],
        prediction: Dict[str, Any],
        ttl: int = CACHE_TTL_SECONDS,
    ) -> bool:
        """
        Stores prediction in Redis cache with TTL (1 hour default).
        """
        r = self._get_redis()
        if r is None:
            return False

        try:
            img_hash = self.compute_image_hash(image_input)
            cache_key = f"dermai:pred:{img_hash}"
            # Copy prediction dict and remove non-serializable fields if any
            clean_dict = {
                k: v for k, v in prediction.items()
                if k not in ("model", "device")
            }
            # Clean NumPy types
            clean_json = json.dumps(clean_dict, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
            r.setex(cache_key, ttl, clean_json)
            logger.info(f"[MLService] Cached prediction for image hash {img_hash[:12]} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.debug(f"[MLService] Redis write failed: {e}")
            return False

    def predict(
        self,
        image_input: Union[str, Path, bytes, bytearray, PILImage.Image, np.ndarray],
        run_segmentation: bool = True,
        threshold: Optional[float] = None,
        image_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes prediction with transparent Redis caching:
        1. Checks cache by SHA-256 hash of image bytes.
        2. If hit, returns cached prediction instantly.
        3. If miss, runs inference and stores result in cache for 1 hour.
        4. If Redis is unavailable, transparently falls back to inference without error.
        """
        cached = self.get_cached_prediction(image_input)
        if cached is not None:
            return cached

        # Execute ensemble inference across available checkpoints
        try:
            ens_result = ensemble_predict(image_input)
            num_models_used = ens_result.get("num_models_used", 1)
        except Exception as e:
            logger.warning(f"[MLService] Ensemble prediction failed ({e}), falling back to single model.")
            ens_result = None
            num_models_used = 1

        # Execute inference pipeline (segmentation, Grad-CAM, risk scoring)
        prediction = inference_service.predict(
            image_input=image_input,
            run_segmentation=run_segmentation,
            threshold=threshold,
            image_id=image_id,
        )

        # Merge ensemble soft-voting results if available
        if ens_result:
            prediction["predicted_class"] = ens_result.get("predicted_class", prediction.get("predicted_class"))
            prediction["confidence"] = ens_result.get("confidence", prediction.get("confidence"))
            prediction["probabilities"] = ens_result.get("probabilities", prediction.get("probabilities"))
            prediction["top_n"] = ens_result.get("top_n", prediction.get("top_n"))
            prediction["ensemble"] = bool(num_models_used > 1)

        prediction["num_models_used"] = num_models_used

        # Store in Redis
        self.set_cached_prediction(image_input, prediction, ttl=CACHE_TTL_SECONDS)
        prediction["cached"] = False
        return prediction


ml_service = MLService()
