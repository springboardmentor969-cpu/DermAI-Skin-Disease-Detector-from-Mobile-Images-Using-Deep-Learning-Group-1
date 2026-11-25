"""
DermAI Multi-Model Ensemble Predictor.

Features:
- Discovers and loads ALL classification model checkpoints (*.pt / *.pth) in ml/experiments/.
- Runs forward passes across all discovered models.
- Performs soft-voting ensemble by computing the unweighted average of predicted softmax probability distributions.
- Operates in single-model mode if only 1 checkpoint is found.
- Falls back to deterministic demo mode if 0 checkpoints are found.
- Returns standardized output format:
    {
        "predicted_class": "mel",
        "confidence": 0.8421,
        "probabilities": {"akiec": 0.01, ...},
        "num_models_used": 2
    }
"""

import io
import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("dermai.ensemble_predict")

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Default deterministic demo distribution
DEMO_CLASS_PROBS = {
    "nv": 0.685,
    "bkl": 0.125,
    "mel": 0.095,
    "bcc": 0.045,
    "akiec": 0.025,
    "df": 0.015,
    "vasc": 0.010,
}

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EnsemblePredictor:
    """
    Manages loading and soft-voting inference across multiple deep learning checkpoints.
    """

    def __init__(
        self,
        experiments_dir: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        if experiments_dir is None:
            self.experiments_dir = PROJECT_ROOT / "ml" / "experiments"
        else:
            self.experiments_dir = Path(experiments_dir)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: List[Tuple[nn.Module, str, str]] = []  # (model, arch_name, checkpoint_path)
        self._loaded = False
        self._load_all_models()

    def _discover_checkpoints(self) -> List[Path]:
        """Finds all classification model checkpoints in experiments directory."""
        if not self.experiments_dir.is_dir():
            return []

        checkpoints = []
        for ext in ["*.pt", "*.pth"]:
            for pt_file in self.experiments_dir.rglob(ext):
                # Filter out segmentation models and last checkpoints if best exists
                path_str = str(pt_file).lower()
                if "segment" in path_str or "mask" in path_str:
                    continue
                # Skip last.pt if best.pt exists in the same folder
                if pt_file.name == "last.pt" and (pt_file.parent / "best.pt").is_file():
                    continue
                if pt_file.is_file() and pt_file.stat().st_size > 1024:
                    checkpoints.append(pt_file)

        # De-duplicate files that point to the exact same path
        unique_checkpoints = []
        seen = set()
        for ckpt in checkpoints:
            resolved = ckpt.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_checkpoints.append(ckpt)

        return unique_checkpoints

    def _load_single_model(self, ckpt_path: Path) -> Optional[Tuple[nn.Module, str]]:
        """Loads a model architecture and weights from checkpoint file."""
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model_state = None
            arch = None

            if isinstance(ckpt, dict):
                model_state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model")
                arch = ckpt.get("model_name") or ckpt.get("arch") or ckpt.get("model")
            elif isinstance(ckpt, nn.Module):
                ckpt.to(self.device)
                ckpt.eval()
                return ckpt, ckpt.__class__.__name__

            # Check adjacent config.json if arch name not in checkpoint dict
            if not arch or not isinstance(arch, str):
                cfg_path = ckpt_path.parent / "config.json"
                if cfg_path.is_file():
                    try:
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        arch = cfg.get("model") or cfg.get("model_name")
                    except Exception:
                        pass

            if not arch or not isinstance(arch, str):
                # Fallback heuristic based on folder name
                folder_name = ckpt_path.parent.name.lower()
                if "swin" in folder_name:
                    arch = "swin_tiny_patch4_window7_224"
                elif "efficient" in folder_name:
                    arch = "efficientnet_b0"
                elif "densenet" in folder_name:
                    arch = "densenet121"
                else:
                    arch = "resnet50"

            import timm
            model = timm.create_model(arch, pretrained=False, num_classes=len(CLASS_NAMES))
            if isinstance(model_state, dict):
                model.load_state_dict(model_state, strict=False)
            model.to(self.device)
            model.eval()
            return model, arch
        except Exception as e:
            logger.warning(f"Could not load checkpoint {ckpt_path}: {e}")
            return None

    def _load_all_models(self) -> None:
        """Discovers and loads all classification checkpoints into memory."""
        checkpoints = self._discover_checkpoints()
        self.models = []

        for ckpt in checkpoints:
            loaded = self._load_single_model(ckpt)
            if loaded:
                model, arch = loaded
                self.models.append((model, arch, str(ckpt)))
                logger.info(f"Loaded ensemble member: {arch} from {ckpt.name}")

        self._loaded = True
        logger.info(f"EnsemblePredictor initialized with {len(self.models)} active model(s).")

    @property
    def num_models_used(self) -> int:
        return len(self.models)

    def _preprocess_input(
        self,
        image_input: Union[str, Path, bytes, bytearray, PILImage.Image, np.ndarray],
    ) -> Tuple[torch.Tensor, str]:
        """Converts heterogeneous image inputs to a normalized (1, 3, 224, 224) torch Tensor."""
        pil_img = None
        hash_seed = ""

        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if p.is_file():
                raw_bytes = p.read_bytes()
                hash_seed = hashlib.sha256(raw_bytes).hexdigest()
                pil_img = PILImage.open(p).convert("RGB")
            else:
                raise FileNotFoundError(f"Image file not found: {image_input}")
        elif isinstance(image_input, (bytes, bytearray)):
            raw_bytes = bytes(image_input)
            hash_seed = hashlib.sha256(raw_bytes).hexdigest()
            pil_img = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")
        elif isinstance(image_input, PILImage.Image):
            pil_img = image_input.convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            hash_seed = hashlib.sha256(buf.getvalue()).hexdigest()
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                image_input = np.stack([image_input] * 3, axis=-1)
            elif image_input.shape[2] == 4:
                image_input = image_input[:, :, :3]
            pil_img = PILImage.fromarray(image_input.astype(np.uint8)).convert("RGB")
            hash_seed = hashlib.sha256(image_input.tobytes()).hexdigest()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        tensor = _TRANSFORM(pil_img).unsqueeze(0).to(self.device)
        return tensor, hash_seed

    def predict(
        self,
        image_input: Union[str, Path, bytes, bytearray, PILImage.Image, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Executes soft-voting ensemble inference across all loaded models.

        Returns:
            dict: {
                "predicted_class": str,
                "confidence": float,
                "probabilities": Dict[str, float],
                "num_models_used": int,
                "top_n": List[Dict[str, Any]]
            }
        """
        tensor, hash_seed = self._preprocess_input(image_input)

        # Case 0: No models found -> Deterministic demo mode
        if len(self.models) == 0:
            logger.info("EnsemblePredictor running in deterministic demo mode (0 models found).")
            # Deterministically permute/scale base probabilities using image hash
            seed_int = int(hash_seed[:8], 16) if hash_seed else 42
            rng = np.random.RandomState(seed_int)

            probs_arr = np.array([DEMO_CLASS_PROBS[c] for c in CLASS_NAMES], dtype=np.float64)
            noise = rng.normal(0, 0.02, size=len(probs_arr))
            probs_arr = np.clip(probs_arr + noise, 0.001, 1.0)
            probs_arr = probs_arr / np.sum(probs_arr)

            top_idx = int(np.argmax(probs_arr))
            predicted_class = CLASS_NAMES[top_idx]
            confidence = round(float(probs_arr[top_idx]), 4)
            probabilities = {CLASS_NAMES[i]: round(float(probs_arr[i]), 4) for i in range(len(CLASS_NAMES))}

            sorted_top_n = sorted(
                [{"class": c, "probability": probabilities[c]} for c in CLASS_NAMES],
                key=lambda x: x["probability"],
                reverse=True,
            )

            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities,
                "num_models_used": 0,
                "top_n": sorted_top_n,
            }

        # Case 1 & 2: 1 or more models found -> Soft voting
        all_model_probs = []

        with torch.no_grad():
            for model, arch, ckpt_str in self.models:
                try:
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    all_model_probs.append(probs)
                except Exception as e:
                    logger.warning(f"Inference failed on model {arch}: {e}")

        if not all_model_probs:
            # Fallback if forward pass threw on all models
            return self.predict_demo_mode(hash_seed)

        # Average probabilities across all successfully evaluated models
        avg_probs = np.mean(all_model_probs, axis=0)
        avg_probs = avg_probs / np.sum(avg_probs)

        top_idx = int(np.argmax(avg_probs))
        predicted_class = CLASS_NAMES[top_idx]
        confidence = round(float(avg_probs[top_idx]), 4)

        probabilities = {
            CLASS_NAMES[i]: round(float(avg_probs[i]), 4)
            for i in range(len(CLASS_NAMES))
        }

        sorted_top_n = sorted(
            [{"class": c, "probability": probabilities[c]} for c in CLASS_NAMES],
            key=lambda x: x["probability"],
            reverse=True,
        )

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "num_models_used": len(all_model_probs),
            "top_n": sorted_top_n,
        }

    def predict_demo_mode(self, hash_seed: str = "") -> Dict[str, Any]:
        """Generates deterministic mock prediction."""
        seed_int = int(hash_seed[:8], 16) if hash_seed else 42
        rng = np.random.RandomState(seed_int)
        probs_arr = np.array([DEMO_CLASS_PROBS[c] for c in CLASS_NAMES], dtype=np.float64)
        noise = rng.normal(0, 0.02, size=len(probs_arr))
        probs_arr = np.clip(probs_arr + noise, 0.001, 1.0)
        probs_arr = probs_arr / np.sum(probs_arr)

        top_idx = int(np.argmax(probs_arr))
        predicted_class = CLASS_NAMES[top_idx]
        confidence = round(float(probs_arr[top_idx]), 4)
        probabilities = {CLASS_NAMES[i]: round(float(probs_arr[i]), 4) for i in range(len(CLASS_NAMES))}

        sorted_top_n = sorted(
            [{"class": c, "probability": probabilities[c]} for c in CLASS_NAMES],
            key=lambda x: x["probability"],
            reverse=True,
        )

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "num_models_used": 0,
            "top_n": sorted_top_n,
        }


# Global singleton instance for clean reusable import
_ENSEMBLE_PREDICTOR: Optional[EnsemblePredictor] = None


def get_ensemble_predictor(experiments_dir: Optional[Union[str, Path]] = None) -> EnsemblePredictor:
    global _ENSEMBLE_PREDICTOR
    if _ENSEMBLE_PREDICTOR is None:
        _ENSEMBLE_PREDICTOR = EnsemblePredictor(experiments_dir=experiments_dir)
    return _ENSEMBLE_PREDICTOR


def ensemble_predict(
    image_input: Union[str, Path, bytes, bytearray, PILImage.Image, np.ndarray],
    experiments_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Convenience function for running multi-model soft-voting ensemble inference.
    """
    predictor = get_ensemble_predictor(experiments_dir=experiments_dir)
    return predictor.predict(image_input)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DermAI Ensemble Prediction CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    args = parser.parse_args()

    predictor = get_ensemble_predictor()
    print(f"Loaded {predictor.num_models_used} model(s) for ensemble:")
    for _, arch, path in predictor.models:
        print(f"  - {arch}: {path}")

    if args.image and Path(args.image).is_file():
        res = predictor.predict(args.image)
        print("\nPrediction Results:")
        print(json.dumps(res, indent=2))
    else:
        # Test with random noise image
        dummy_img = PILImage.new("RGB", (224, 224), color=(180, 120, 80))
        res = predictor.predict(dummy_img)
        print("\nDummy Test Image Prediction Results:")
        print(json.dumps(res, indent=2))
