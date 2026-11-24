"""
DermAI Model Inference Service

Loads trained DermAI ResNet-50 and DenseNet-121 checkpoints
and performs image classification.

Project structure expected:

DermAI/
│
├── backend/
│   └── services/
│       └── inference.py
│
├── models/
│   ├── classification/
│   │   ├── resnet50.py
│   │   └── densenet121.py
│   │
│   └── checkpoints/
│       ├── resnet50_best.pth
│       └── densenet121_best.pth
│
└── processed_images/
"""

from pathlib import Path
import importlib.util
import sys

import torch
from PIL import Image
from torchvision import transforms
from backend.services.risk_assessment import assess_risk

# ============================================================
# 1. DERMAI ROOT
# ============================================================

# inference.py:
#
# DermAI/
#   backend/
#       services/
#           inference.py
#
# parents[0] = services
# parents[1] = backend
# parents[2] = DermAI

DERMAI_ROOT = Path(__file__).resolve().parents[2]

MODELS_ROOT = DERMAI_ROOT / "models"
CLASSIFICATION_ROOT = MODELS_ROOT / "classification"
CHECKPOINT_ROOT = MODELS_ROOT / "checkpoints"


# ============================================================
# 2. VERIFY REQUIRED DIRECTORIES
# ============================================================

if not DERMAI_ROOT.exists():
    raise RuntimeError(
        f"DermAI root directory does not exist:\n{DERMAI_ROOT}"
    )

if not CLASSIFICATION_ROOT.exists():
    raise RuntimeError(
        f"Classification model directory does not exist:\n"
        f"{CLASSIFICATION_ROOT}"
    )

if not CHECKPOINT_ROOT.exists():
    raise RuntimeError(
        f"Checkpoint directory does not exist:\n"
        f"{CHECKPOINT_ROOT}"
    )


# ============================================================
# 3. DIRECT MODEL FILE PATHS
# ============================================================

RESNET50_FILE = CLASSIFICATION_ROOT / "resnet50.py"
DENSENET121_FILE = CLASSIFICATION_ROOT / "densenet121.py"


if not RESNET50_FILE.exists():
    raise FileNotFoundError(
        f"ResNet-50 model file not found:\n{RESNET50_FILE}"
    )

if not DENSENET121_FILE.exists():
    raise FileNotFoundError(
        f"DenseNet-121 model file not found:\n{DENSENET121_FILE}"
    )


# ============================================================
# 4. LOAD MODEL FACTORY DIRECTLY FROM FILE
# ============================================================

def _load_module_from_file(module_name, file_path):
    """
    Load a Python module directly from its absolute file path.

    This avoids depending on Python resolving the top-level
    package name 'models', which can cause conflicts when
    Uvicorn reload/spawn processes are used.
    """

    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(
            f"Python module file not found:\n{file_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(file_path)
    )

    if spec is None:
        raise ImportError(
            f"Could not create import specification for:\n{file_path}"
        )

    if spec.loader is None:
        raise ImportError(
            f"Could not create module loader for:\n{file_path}"
        )

    module = importlib.util.module_from_spec(spec)

    # Register module before execution.
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


# Load the actual DermAI model files.
_resnet_module = _load_module_from_file(
    "dermai_resnet50_model",
    RESNET50_FILE
)

_densenet_module = _load_module_from_file(
    "dermai_densenet121_model",
    DENSENET121_FILE
)


# Get factory functions.
create_resnet50 = _resnet_module.create_resnet50
create_densenet121 = _densenet_module.create_densenet121


# ============================================================
# 5. CHECKPOINT PATHS
# ============================================================

RESNET50_CHECKPOINT = (
    CHECKPOINT_ROOT / "resnet50_best.pth"
)

DENSENET121_CHECKPOINT = (
    CHECKPOINT_ROOT / "densenet121_best.pth"
)


# ============================================================
# 6. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 7. CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
]


NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# 8. IMAGE PREPROCESSING FOR INFERENCE
# ============================================================

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# 9. MODEL CACHE
# ============================================================

_MODEL_CACHE = {}


# ============================================================
# 10. CHECKPOINT STATE DICT EXTRACTION
# ============================================================

def _extract_state_dict(checkpoint):
    """
    Extract a model state dictionary from common checkpoint formats.
    """

    if isinstance(checkpoint, dict):

        # Direct state dictionary
        if all(
            isinstance(key, str)
            for key in checkpoint.keys()
        ):
            tensor_values = [
                value
                for value in checkpoint.values()
                if torch.is_tensor(value)
            ]

            if len(tensor_values) > 0:
                return checkpoint

        # Common training checkpoint format
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

    raise RuntimeError(
        "Could not find a valid model state dictionary "
        "inside the checkpoint."
    )


# ============================================================
# 11. CLEAN STATE DICT
# ============================================================

def _clean_state_dict(state_dict):
    """
    Remove 'module.' prefix if checkpoint was created
    using DataParallel.
    """

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key

        cleaned[new_key] = value

    return cleaned


# ============================================================
# 12. CREATE MODEL
# ============================================================

def _create_model(model_name):
    """
    Create an uninitialized DermAI model architecture.
    """

    normalized_name = model_name.lower().strip()

    if normalized_name in {
        "resnet50",
        "resnet-50",
        "resnet_50",
    }:

        model = create_resnet50(
            num_classes=NUM_CLASSES,
            pretrained=False,
        )

        return model

    if normalized_name in {
        "densenet121",
        "densenet-121",
        "densenet_121",
    }:

        model = create_densenet121(
            num_classes=NUM_CLASSES,
            pretrained=False,
        )

        return model

    raise ValueError(
        f"Unsupported model: {model_name}. "
        f"Available models: resnet50, densenet121"
    )


# ============================================================
# 13. CHECKPOINT PATH
# ============================================================

def _get_checkpoint_path(model_name):
    """
    Return the checkpoint corresponding to the selected model.
    """

    normalized_name = model_name.lower().strip()

    if normalized_name in {
        "resnet50",
        "resnet-50",
        "resnet_50",
    }:
        return RESNET50_CHECKPOINT

    if normalized_name in {
        "densenet121",
        "densenet-121",
        "densenet_121",
    }:
        return DENSENET121_CHECKPOINT

    raise ValueError(
        f"Unsupported model: {model_name}"
    )


# ============================================================
# 14. LOAD MODEL
# ============================================================

def load_model(model_name):
    """
    Load a trained DermAI model.

    The model is cached after the first load.
    """

    normalized_name = model_name.lower().strip()

    # Return cached model.
    if normalized_name in _MODEL_CACHE:
        return _MODEL_CACHE[normalized_name]

    checkpoint_path = _get_checkpoint_path(
        normalized_name
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {normalized_name}:\n"
            f"{checkpoint_path}"
        )

    print(
        f"[DermAI] Loading model: {normalized_name}"
    )

    print(
        f"[DermAI] Checkpoint: {checkpoint_path}"
    )

    print(
        f"[DermAI] Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Create architecture
    # --------------------------------------------------------

    model = _create_model(
        normalized_name
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )

    except TypeError:

        # Compatibility fallback
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    # --------------------------------------------------------
    # Extract state dict
    # --------------------------------------------------------

    state_dict = _extract_state_dict(
        checkpoint
    )

    state_dict = _clean_state_dict(
        state_dict
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    # --------------------------------------------------------
    # Move to device
    # --------------------------------------------------------

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    _MODEL_CACHE[normalized_name] = model

    print(
        f"[DermAI] {normalized_name} loaded successfully."
    )

    return model


# ============================================================
# 15. PREPARE IMAGE
# ============================================================

def _prepare_image(image):
    """
    Convert an input PIL image into a model-ready tensor.
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            "Expected a PIL.Image.Image object."
        )

    # Ensure RGB.
    image = image.convert("RGB")

    # Apply inference preprocessing.
    tensor = INFERENCE_TRANSFORM(image)

    # Add batch dimension.
    tensor = tensor.unsqueeze(0)

    # Move to CPU/GPU.
    tensor = tensor.to(DEVICE)

    return tensor


# ============================================================
# 16. BUILD TOP PREDICTIONS
# ============================================================

def _build_top_predictions(probabilities):
    """
    Build sorted prediction list.
    """

    results = []

    for index, probability in enumerate(
        probabilities
    ):

        results.append(
            {
                "condition": CLASS_NAMES[index],
                "probability": float(probability),
                "confidence": float(probability),
            }
        )

    results.sort(
        key=lambda item: item["probability"],
        reverse=True,
    )

    return results


# ============================================================
# 17. PREDICT IMAGE
# ============================================================

def predict_image(
    image,
    model_name="resnet50",
):
    """
    Run DermAI classification on one image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input dermatological image.

    model_name : str
        'resnet50' or 'densenet121'

    Returns
    -------
    dict
        Prediction information.
    """

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        model_name
    )

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    input_tensor = _prepare_image(
        image
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

    # --------------------------------------------------------
    # Handle models returning tuples/lists
    # --------------------------------------------------------

    if isinstance(output, (tuple, list)):

        output = output[0]

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if output.ndim != 2:

        raise RuntimeError(
            f"Unexpected model output shape: "
            f"{tuple(output.shape)}"
        )

    if output.shape[1] != NUM_CLASSES:

        raise RuntimeError(
            f"Model returned {output.shape[1]} classes, "
            f"but DermAI expects {NUM_CLASSES}."
        )

    # --------------------------------------------------------
    # Softmax
    # --------------------------------------------------------

    probabilities = torch.softmax(
        output,
        dim=1,
    )[0]

    probabilities = (
        probabilities
        .detach()
        .cpu()
        .tolist()
    )

    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    predicted_index = int(
        torch.argmax(
            torch.tensor(probabilities)
        ).item()
    )

    predicted_condition = (
        CLASS_NAMES[predicted_index]
    )

    predicted_confidence = float(
        probabilities[predicted_index]
    )

    # --------------------------------------------------------
    # Risk assessment
    # --------------------------------------------------------

    risk_assessment = assess_risk(
        condition=predicted_condition,
        confidence=predicted_confidence
    )

    # --------------------------------------------------------
    # Top predictions
    # --------------------------------------------------------

    top_predictions = _build_top_predictions(
        probabilities
    )

    # --------------------------------------------------------
    # Original image size
    # --------------------------------------------------------

    image_width, image_height = image.size

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "condition": predicted_condition,

        "confidence": predicted_confidence,

        "probability": predicted_confidence,

        "top_predictions": top_predictions,

        "risk_assessment": risk_assessment,

        "risk_category": risk_assessment["risk_category"],

        "review_priority": risk_assessment["review_priority"],

        "assessment_type": risk_assessment["assessment_type"],

        "risk_message": risk_assessment["message"],

        "medical_disclaimer": risk_assessment["disclaimer"],

        "model": model_name,

        "device": str(DEVICE),

        "image_size": {
            "width": image_width,
            "height": image_height,
        },

        "classes": CLASS_NAMES,
    }

# ============================================================
# 18. AVAILABLE MODELS
# ============================================================

def get_available_models():
    """
    Return information about available DermAI models.
    """

    return [
        {
            "name": "resnet50",
            "display_name": "ResNet-50",
            "checkpoint": str(
                RESNET50_CHECKPOINT
            ),
            "checkpoint_exists":
                RESNET50_CHECKPOINT.exists(),
        },
        {
            "name": "densenet121",
            "display_name": "DenseNet-121",
            "checkpoint": str(
                DENSENET121_CHECKPOINT
            ),
            "checkpoint_exists":
                DENSENET121_CHECKPOINT.exists(),
        },
    ]


# ============================================================
# 19. MODEL STATUS
# ============================================================

def get_model_status():
    """
    Return current inference service status.
    """

    return {
        "device": str(DEVICE),

        "dermai_root": str(
            DERMAI_ROOT
        ),

        "models_root": str(
            MODELS_ROOT
        ),

        "classification_root": str(
            CLASSIFICATION_ROOT
        ),

        "checkpoint_root": str(
            CHECKPOINT_ROOT
        ),

        "classes": CLASS_NAMES,

        "num_classes": NUM_CLASSES,

        "models": get_available_models(),

        "loaded_models": list(
            _MODEL_CACHE.keys()
        ),
    }


# ============================================================
# 20. CLEAR MODEL CACHE
# ============================================================

def clear_model_cache():
    """
    Remove all cached models.
    """

    global _MODEL_CACHE

    _MODEL_CACHE.clear()

    # Release CUDA cache if available.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "[DermAI] Model cache cleared."
    )


# ============================================================
# 21. STARTUP INFORMATION
# ============================================================

print("=" * 60)
print("DERMAI INFERENCE SERVICE")
print("=" * 60)
print(
    f"DermAI root       : {DERMAI_ROOT}"
)
print(
    f"Classification    : {CLASSIFICATION_ROOT}"
)
print(
    f"Checkpoints       : {CHECKPOINT_ROOT}"
)
print(
    f"ResNet-50 file    : {RESNET50_FILE}"
)
print(
    f"DenseNet-121 file : {DENSENET121_FILE}"
)
print(
    f"Device             : {DEVICE}"
)
print(
    f"Classes            : {CLASS_NAMES}"
)
print("=" * 60)