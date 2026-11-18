
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ==========================================
# 1. SETTINGS
# ==========================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "dataset", "images")
RESULTS = os.path.join(ROOT, "results")

os.makedirs(RESULTS, exist_ok=True)

DEVICE = torch.device("cpu")

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

LIMIT = 1200
BATCH_SIZE = 8

torch.manual_seed(42)
np.random.seed(42)

print("DERMAI CONFIDENCE ESTIMATION STARTED")


# ==========================================
# 2. FIND IMAGES
# ==========================================

IMAGE_LOOKUP = {}

for folder, subfolders, files in os.walk(DATASET):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg")):
            image_id = os.path.splitext(filename)[0]
            IMAGE_LOOKUP[image_id] = os.path.join(folder, filename)

print("Images found:", len(IMAGE_LOOKUP))

if not IMAGE_LOOKUP:
    raise FileNotFoundError("No dataset images found.")


# ==========================================
# 3. LOAD METADATA
# ==========================================

csv_files = glob.glob(
    os.path.join(DATASET, "**", "*.csv"),
    recursive=True
)

metadata_path = None

for path in csv_files:
    try:
        columns = pd.read_csv(path, nrows=0).columns
        if "image_id" in columns and "dx" in columns:
            metadata_path = path
            break
    except Exception:
        continue

if metadata_path is None:
    raise FileNotFoundError("HAM10000 metadata CSV not found.")

df = pd.read_csv(metadata_path)

df.columns = df.columns.astype(str).str.strip()
df["image_id"] = df["image_id"].astype(str).str.strip()
df["dx"] = df["dx"].astype(str).str.strip().str.lower()

df = df.dropna(subset=["image_id", "dx"])
df = df[df["dx"].isin(CLASSES)]
df = df[df["image_id"].isin(IMAGE_LOOKUP)]
df = df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)

if len(df) == 0:
    raise ValueError("No matching images and metadata records.")

if df["dx"].value_counts().min() < 2:
    raise ValueError("Not enough images for stratified splitting.")

# Reproduce the HPO sampling and validation split
df = df.sample(
    n=min(LIMIT, len(df)),
    random_state=42
).reset_index(drop=True)

_, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["dx"]
)

print("Images selected for evaluation:", len(test_df))


# ==========================================
# 4. IMAGE TRANSFORM
# ==========================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# 5. DATASET
# ==========================================

class HAMDataset(Dataset):

    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):

        row = self.frame.iloc[index]

        image_id = row["image_id"]
        image_path = IMAGE_LOOKUP[image_id]

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = image_transform(image)

        label = CLASSES.index(row["dx"])

        return image, label, image_id


loader = DataLoader(
    HAMDataset(test_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 6. LOAD TRAINED MODELS
# ==========================================

def load_checkpoint(filename):

    path = os.path.join(RESULTS, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Checkpoint not found: " + path
        )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    saved_classes = checkpoint.get("classes")

    if saved_classes is not None:
        if list(saved_classes) != CLASSES:
            raise ValueError(
                "Class order mismatch: " + filename
            )

    return checkpoint["model_state_dict"]


# ResNet18
resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, 7)

resnet.load_state_dict(
    load_checkpoint("resnet18_comparison.pth")
)


# MobileNetV3-Small
mobile = models.mobilenet_v3_small(weights=None)
mobile.classifier[3] = nn.Linear(
    mobile.classifier[3].in_features, 7
)

mobile.load_state_dict(
    load_checkpoint("mobilenet_v3_small.pth")
)


# EfficientNet-B0
efficient = models.efficientnet_b0(weights=None)
efficient.classifier[1] = nn.Linear(
    efficient.classifier[1].in_features, 7
)

efficient.load_state_dict(
    load_checkpoint("dermai_efficientnet_b0.pth")
)

models_list = [resnet, mobile, efficient]

for model in models_list:
    model.to(DEVICE)
    model.eval()

print("All three models loaded successfully.")


# ==========================================
# 7. CONFIDENCE AND UNCERTAINTY
# ==========================================

prediction_rows = []

with torch.no_grad():

    for images, labels, image_ids in loader:

        images = images.to(DEVICE)

        probabilities = []

        for model in models_list:

            outputs = model(images)

            probabilities.append(
                torch.softmax(outputs, dim=1)
            )

        # Average model scores
        ensemble_probs = torch.stack(
            probabilities
        ).mean(dim=0)

        # Sort scores from highest to lowest
        sorted_probs, sorted_indices = torch.sort(
            ensemble_probs,
            dim=1,
            descending=True
        )

        for i, image_id in enumerate(image_ids):

            top_score = float(sorted_probs[i, 0])
            second_score = float(sorted_probs[i, 1])

            predicted_index = int(sorted_indices[i, 0])

            # Difference between first and second scores
            margin = top_score - second_score

            # Normalized prediction entropy
            probabilities_row = ensemble_probs[i].cpu().numpy()

            entropy = -np.sum(
                probabilities_row *
                np.log(np.clip(probabilities_row, 1e-12, 1.0))
            )

            normalized_entropy = entropy / np.log(len(CLASSES))

            prediction_rows.append({

                "image_id": image_id,

                "actual_class": CLASSES[int(labels[i])],

                "predicted_class": CLASSES[predicted_index],

                "top_score": round(top_score, 4),

                "second_score": round(second_score, 4),

                "prediction_margin": round(margin, 4),

                "normalized_entropy": round(
                    float(normalized_entropy), 4
                )
            })


# ==========================================
# 8. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(prediction_rows)

output_path = os.path.join(
    RESULTS,
    "confidence_estimation_results.csv"
)

results_df.to_csv(output_path, index=False)


# ==========================================
# 9. SUMMARY
# ==========================================

accuracy = (
    results_df["actual_class"] ==
    results_df["predicted_class"]
).mean() * 100

print("\n====================================")
print("CONFIDENCE ESTIMATION COMPLETED")
print("====================================")

print("Images evaluated:", len(results_df))
print("Ensemble accuracy:", round(accuracy, 2), "%")

print(
    "Average top score:",
    round(results_df["top_score"].mean() * 100, 2),
    "%"
)

print(
    "Average prediction margin:",
    round(results_df["prediction_margin"].mean(), 4)
)

print(
    "Average normalized entropy:",
    round(results_df["normalized_entropy"].mean(), 4)
)

print("\nResults saved:")
print(output_path)

print("\nScores are uncalibrated model outputs.")
print("They must not be interpreted as medical risk.")
