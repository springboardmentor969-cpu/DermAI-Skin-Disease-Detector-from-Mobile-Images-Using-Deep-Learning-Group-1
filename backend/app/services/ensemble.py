"""
DermAI Milestone 2: Multi-Model Ensemble Inference Service.

Implements soft-voting ensemble across multiple deep learning models:
1. Loads each model architecture and weights from checkpoint/config.json.
2. Runs forward pass on the identical preprocessed image tensor.
3. Computes and averages softmax probability distributions across all models.
4. Returns the standardized prediction response format with top prediction,
   all class probabilities, and confidence score.
5. Falls back seamlessly to single-model inference if fewer than 2 valid model paths are provided.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import timm

# Ensure project root is available
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings

logger = logging.getLogger("dermai.ensemble")

# Standard HAM10000 7-class dermatological diagnostic mapping
DEFAULT_CLASSES: Dict[str, int] = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}
DEFAULT_IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in DEFAULT_CLASSES.items()}

CLASS_LABELS: Dict[str, str] = {
    "akiec": "Actinic Keratosis / Bowen's Disease",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}

# In-memory cache for loaded ensemble models: (path_str, device_str) -> (model, model_name, idx_to_class)
_MODEL_CACHE: Dict[Tuple[str, str], Tuple[nn.Module, str, Dict[int, str]]] = {}


def resolve_model_path(path_str: Union[str, Path]) -> Optional[Path]:
    """Resolves an ensemble model path candidate, checking relative and absolute locations."""
    p = Path(path_str)
    candidates = [
        p,
        PROJECT_ROOT / p,
        BACKEND_DIR / p,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
        if c.is_dir() and (c / "best.pt").is_file():
            return (c / "best.pt").resolve()
        if c.is_dir() and (c / "last.pt").is_file():
            return (c / "last.pt").resolve()
    return None


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, str, Dict[int, str]]:
    """
    Loads a model from a checkpoint, dynamically identifying its architecture
    from checkpoint metadata or adjacent config.json.
    """
    cache_key = (str(checkpoint_path.resolve()), str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 1. Detect architecture name
    model_name: Optional[str] = None
    classes_map: Dict[str, int] = DEFAULT_CLASSES

    if isinstance(ckpt, dict):
        model_name = ckpt.get("model_name") or ckpt.get("model")
        if not model_name and "config" in ckpt and isinstance(ckpt["config"], dict):
            model_name = ckpt["config"].get("model") or ckpt["config"].get("model_name")
        if "classes" in ckpt and isinstance(ckpt["classes"], dict):
            classes_map = ckpt["classes"]

    # If not in checkpoint dict, check adjacent config.json
    if not model_name:
        config_path = checkpoint_path.parent / "config.json"
        if config_path.is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    model_name = cfg.get("model") or cfg.get("model_name") or cfg.get("architecture")
                    if "classes" in cfg and isinstance(cfg["classes"], dict):
                        classes_map = cfg["classes"]
            except Exception as e:
                logger.warning(f"Could not read adjacent config.json at {config_path}: {e}")

    # Fallback to resnet50 if unidentified
    if not model_name:
        model_name = "resnet50"
        logger.warning(f"Model architecture not specified for {checkpoint_path}. Defaulting to 'resnet50'.")

    idx_to_class = {v: k for k, v in classes_map.items()}

    # 2. Extract and sanitize state_dict
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    clean_state_dict = {}
    if isinstance(state_dict, dict):
        for k, v in state_dict.items():
            clean_k = k[7:] if k.startswith("module.") else k
            clean_state_dict[clean_k] = v
    else:
        clean_state_dict = state_dict

    # 3. Instantiate model via timm
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(classes_map),
    )
    if isinstance(clean_state_dict, dict):
        model.load_state_dict(clean_state_dict, strict=False)

    model.to(device)
    model.eval()

    logger.info(
        f"[EnsembleService] Loaded '{model_name}' ({len(classes_map)} classes) "
        f"from {checkpoint_path} onto {device}."
    )

    result = (model, model_name, idx_to_class)
    _MODEL_CACHE[cache_key] = result
    return result


def ensemble_predict(
    image_tensor: Union[torch.Tensor, np.ndarray],
    model_paths: List[str],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Executes soft-voting ensemble prediction across multiple model checkpoints.

    Args:
        image_tensor: Normalized image tensor of shape (1, 3, 224, 224) or (3, 224, 224).
        model_paths: List of file paths to model checkpoints.
        device: PyTorch device to execute inference on (defaults to cuda if available).

    Returns:
        Dictionary containing:
            - predicted_class: string diagnostic code (e.g. 'mel')
            - predicted_class_name: human-readable name (e.g. 'Melanoma')
            - confidence: float top confidence probability
            - top_n: list of top 3 condition predictions
            - all_predictions: list of all 7 condition predictions with probabilities
            - probabilities: dict of class_code -> float probability
            - models_used: list of model architecture names / paths used
            - ensemble: bool indicating whether multi-model ensemble was active
            - model_version: descriptive model version string
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Format input tensor to (1, C, H, W)
    if isinstance(image_tensor, np.ndarray):
        tensor = torch.from_numpy(image_tensor).float()
    else:
        tensor = image_tensor.float()

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    # 1. Filter and validate model paths
    resolved_paths: List[Path] = []
    for p in model_paths:
        res = resolve_model_path(p)
        if res and res not in resolved_paths:
            resolved_paths.append(res)
        elif not res:
            logger.warning(f"[EnsembleService] Model path not found: {p}")

    # 2. Fallback to single model if fewer than 2 valid paths
    if len(resolved_paths) < 2:
        logger.info(
            f"[EnsembleService] Fewer than 2 valid models provided (found {len(resolved_paths)}). "
            f"Falling back to single model inference."
        )
        single_path = resolved_paths[0] if len(resolved_paths) == 1 else resolve_model_path(settings.MODEL_PATH)
        if not single_path:
            raise FileNotFoundError(
                f"No valid model checkpoint available for inference (tried {model_paths} and {settings.MODEL_PATH})."
            )

        model, model_name, idx_to_class = load_model_from_checkpoint(single_path, device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        ranked_indices = np.argsort(probs)[::-1]
        all_preds = []
        prob_dict = {}
        for idx in ranked_indices:
            code = idx_to_class.get(int(idx), f"class_{idx}")
            label = CLASS_LABELS.get(code, code.upper())
            prob_val = float(round(float(probs[idx]), 4))
            prob_dict[code] = prob_val
            all_preds.append({"class": code, "label": label, "probability": prob_val})

        top_pred = all_preds[0]
        return {
            "predicted_class": top_pred["class"],
            "predicted_class_name": top_pred["label"],
            "confidence": top_pred["probability"],
            "top_n": all_preds[:3],
            "all_predictions": all_preds,
            "probabilities": prob_dict,
            "models_used": [model_name],
            "ensemble": False,
            "model_version": model_name,
        }

    # 3. Multi-model Ensemble Execution
    loaded_models: List[Tuple[nn.Module, str, Dict[int, str]]] = []
    for path in resolved_paths:
        try:
            m = load_model_from_checkpoint(path, device)
            loaded_models.append(m)
        except Exception as e:
            logger.error(f"[EnsembleService] Failed to load model from {path}: {e}")

    if len(loaded_models) < 2:
        logger.warning("[EnsembleService] Could not load at least 2 models successfully. Falling back to single model.")
        if len(loaded_models) == 1:
            model, model_name, idx_to_class = loaded_models[0]
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            ranked_indices = np.argsort(probs)[::-1]
            all_preds = [
                {
                    "class": idx_to_class.get(int(i), f"class_{i}"),
                    "label": CLASS_LABELS.get(idx_to_class.get(int(i)), str(i)),
                    "probability": float(round(float(probs[i]), 4)),
                }
                for i in ranked_indices
            ]
            return {
                "predicted_class": all_preds[0]["class"],
                "predicted_class_name": all_preds[0]["label"],
                "confidence": all_preds[0]["probability"],
                "top_n": all_preds[:3],
                "all_predictions": all_preds,
                "probabilities": {p["class"]: p["probability"] for p in all_preds},
                "models_used": [model_name],
                "ensemble": False,
                "model_version": model_name,
            }
        else:
            raise RuntimeError("Failed to load any ensemble models.")

    # 4. Soft Voting: Run forward passes and average softmax probability distributions
    model_probs_list: List[np.ndarray] = []
    models_used: List[str] = []

    for model, model_name, _ in loaded_models:
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            model_probs_list.append(probs)
            models_used.append(model_name)

    # Average probabilities across all ensemble members
    avg_probs = np.mean(np.array(model_probs_list), axis=0)

    # Common class mapping
    idx_to_class = loaded_models[0][2]

    # Rank predictions
    ranked_indices = np.argsort(avg_probs)[::-1]
    all_predictions = []
    prob_dict = {}

    for idx in ranked_indices:
        cls_code = idx_to_class.get(int(idx), f"class_{idx}")
        label = CLASS_LABELS.get(cls_code, cls_code.upper())
        p_val = float(round(float(avg_probs[idx]), 4))
        prob_dict[cls_code] = p_val
        all_predictions.append({
            "class": cls_code,
            "label": label,
            "probability": p_val,
        })

    top_pred = all_predictions[0]
    version_tag = f"ensemble({'+'.join(models_used)})"

    return {
        "predicted_class": top_pred["class"],
        "predicted_class_name": top_pred["label"],
        "confidence": top_pred["probability"],
        "top_n": all_predictions[:3],
        "all_predictions": all_predictions,
        "probabilities": prob_dict,
        "models_used": models_used,
        "ensemble": True,
        "model_version": version_tag,
    }

