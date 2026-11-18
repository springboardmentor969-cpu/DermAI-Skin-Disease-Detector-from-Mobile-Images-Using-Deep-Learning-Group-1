import io
import csv
import json
import base64
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, ImageStat, ImageDraw

import torch
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATASET = ROOT / "dataset"
STATIC = Path(__file__).parent / "static"
HISTORY_FILE = RESULTS / "prediction_history.json"

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MODEL_FILE = RESULTS / "mobilenet_v3_small.pth"

app = FastAPI(title="DermAI Milestone 3")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

net = None
model_status = "Model not loaded"

def load_mobilenet():
    global net, model_status
    try:
        if not MODEL_FILE.exists():
            model_status = f"Model file missing: {MODEL_FILE.name}"
            return
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(CLASSES))
        state = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)

        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]

        if not isinstance(state, dict):
            raise ValueError("Unsupported checkpoint format")

        state = {key.replace("module.", ""): value for key, value in state.items()}
        model.load_state_dict(state)
        model.eval()
        net = model
        model_status = "MobileNetV3 loaded"
    except Exception as exc:
        net = None
        model_status = "Model load issue: " + str(exc)

load_mobilenet()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def image_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded

def read_history():
    try:
        if HISTORY_FILE.exists():
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []

def save_history(records):
    RESULTS.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(records[:200], indent=2), encoding="utf-8")


def weighted_f1_from_csv(filename):
    path = RESULTS / filename

    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        if not rows:
            return None

        for row in rows:
            values = {
                str(key).strip().lower(): value
                for key, value in row.items()
                if key is not None
            }

            class_name = (
                values.get("class")
                or values.get("")
                or values.get("h1")
            )

            f1_value = values.get("f1") or values.get("f1-score")
            support_value = values.get("support")

            if (
                class_name
                and class_name.strip().lower() == "weighted avg"
                and f1_value is not None
            ):
                return float(f1_value)

        weighted_total = 0.0
        total_support = 0.0

        for row in rows:
            values = {
                str(key).strip().lower(): value
                for key, value in row.items()
                if key is not None
            }

            class_name = (
                values.get("class")
                or values.get("")
                or values.get("h1")
            )

            f1_value = values.get("f1") or values.get("f1-score")
            support_value = values.get("support")

            if not class_name or class_name.strip() not in CLASSES:
                continue

            if f1_value is None or support_value is None:
                continue

            f1 = float(f1_value)
            support = float(support_value)

            weighted_total += f1 * support
            total_support += support

        if total_support > 0:
            return weighted_total / total_support

    except Exception as error:
        print(f"Could not read {filename}: {error}")

    return None


MODEL_REPORTS = [
    ("Baseline", "baseline_classification_report.csv"),
    ("ResNet18 Transfer Learning", "resnet18_classification_report.csv"),
    ("ResNet18 CNN", "resnet18_comparison_report.csv"),
    ("MobileNetV3-Small", "mobilenet_v3_small_report.csv"),
    ("EfficientNet-B0", "efficientnet_b0_report.csv"),
    ("Vision Transformer", "vision_transformer_report.csv"),
    ("Ensemble", "ensemble_classification_report.csv"),
]

@app.get("/", response_class=HTMLResponse)
def home():
    index_file = STATIC / "index.html"
    if not index_file.exists():
        raise HTTPException(500, "Dashboard file static/index.html is missing.")
    return index_file.read_text(encoding="utf-8")

@app.get("/api/status")
def status():
    preprocessed = DATASET / "preprocessed_images"
    image_count = sum(1 for item in preprocessed.rglob("*") if item.is_file()) if preprocessed.exists() else 0
    return {
        "project": "DermAI Milestone 3",
        "model_status": model_status,
        "model_ready": net is not None,
        "preprocessed_images": image_count,
        "history_count": len(read_history()),
    }

@app.get("/api/metrics")
def metrics():
    model_rows = []
    for model_name, filename in MODEL_REPORTS:
        score = weighted_f1_from_csv(filename)
        if score is not None:
            model_rows.append({
                "model": model_name,
                "weighted_f1": score,
                "weighted_f1_percent": round(score * 100, 2),
                "report_file": filename,
                "note": "Reported weighted F1; evaluation samples/splits may differ."
            })
    best = max(model_rows, key=lambda row: row["weighted_f1"]) if model_rows else None
    return {
        "models": model_rows,
        "best_model_by_weighted_f1": best["model"] if best else None,
        "comparison_warning": "Do not treat these scores as a controlled head-to-head comparison unless all models used the same untouched test split."
    }

 