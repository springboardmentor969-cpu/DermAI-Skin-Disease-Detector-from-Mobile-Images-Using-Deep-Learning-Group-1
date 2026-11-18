
import os
import glob
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


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

print("DERMAI ENSEMBLE MODEL STARTED")


# ==========================================
# 2. INDEX ALL IMAGES
# ==========================================

print("\nSearching for dataset images...")

IMAGE_LOOKUP = {}

for folder, subfolders, files in os.walk(DATASET):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg")):
            image_id = os.path.splitext(filename)[0]
            IMAGE_LOOKUP[image_id] = os.path.join(folder, filename)

print("Images found:", len(IMAGE_LOOKUP))

if not IMAGE_LOOKUP:
    raise FileNotFoundError("No images found in dataset/images.")


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
    raise FileNotFoundError(
        "Metadata CSV containing image_id and dx was not found."
    )

print("Metadata file:", metadata_path)

df = pd.read_csv(metadata_path)

# Clean metadata columns
df.columns = df.columns.astype(str).str.strip()
df["image_id"] = df["image_id"].astype(str).str.strip()
df["dx"] = df["dx"].astype(str).str.strip().str.lower()

df = df.dropna(subset=["image_id", "dx"])
df = df[df["dx"].isin(CLASSES)].copy()

# Keep only images that exist
df = df[df["image_id"].isin(IMAGE_LOOKUP)].copy()

# Remove duplicate image IDs
df = df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)

print("Matching metadata records:", len(df))

if len(df) == 0:
    raise ValueError("No matching metadata and image records found.")

# Check class availability
counts = df["dx"].value_counts()

missing_classes = [c for c in CLASSES if c not in counts.index]

if missing_classes:
    raise ValueError(
        "Missing classes in dataset: " + str(missing_classes)
    )

if counts.min() < 2:
    raise ValueError("Not enough samples for stratified splitting.")


# ==========================================
# 4. STRATIFIED SAMPLE AND SPLIT
# ==========================================

# Sample proportionally without groupby/apply
sample_size = min(LIMIT, len(df))

sampled_parts = []

for class_name in CLASSES:
    class_data = df[df["dx"] == class_name]

    class_n = max(
        1,
        round(sample_size * len(class_data) / len(df))
    )

    class_n = min(class_n, len(class_data))

    sampled_parts.append(
        class_data.sample(
            n=class_n,
            random_state=42
        )
    )

df = pd.concat(sampled_parts, ignore_index=True)

# Trim any extra records while preserving class representation
if len(df) > LIMIT:
    df, _ = train_test_split(
        df,
        train_size=LIMIT,
        random_state=42,
        stratify=df["dx"]
    )

df = df.reset_index(drop=True)

print("Images selected:", len(df))
print("Class counts:")
print(df["dx"].value_counts())

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["dx"]
)

print("Training reference:", len(train_df))
print("Testing images:", len(test_df))


# ==========================================
# 5. IMAGE DATASET
# ==========================================

class HAMDataset(Dataset):

    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):

        row = self.frame.iloc[index]

        image_id = row["image_id"]
        image_path = IMAGE_LOOKUP[image_id]

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = self.transform(image)

        label = CLASSES.index(row["dx"])

        return image, label, image_id


test_loader = DataLoader(
    HAMDataset(test_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 6. LOAD MODEL CHECKPOINTS
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

    if saved_classes is not None and list(saved_classes) != CLASSES:
        raise ValueError(
            "Class order mismatch in " + filename
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

print("\nAll three CNN models loaded successfully.")


# ==========================================
# 7. ENSEMBLE PREDICTION
# ==========================================

actual_labels = []
predicted_labels = []
prediction_rows = []

with torch.no_grad():

    for images, labels, image_ids in test_loader:

        images = images.to(DEVICE)

        probabilities = []

        for model in models_list:

            outputs = model(images)

            probabilities.append(
                torch.softmax(outputs, dim=1)
            )

        # Average probability scores
        ensemble_probs = torch.stack(
            probabilities
        ).mean(dim=0)

        predictions = ensemble_probs.argmax(dim=1)

        actual_labels.extend(labels.tolist())
        predicted_labels.extend(predictions.tolist())

        for i, image_id in enumerate(image_ids):

            predicted_index = int(predictions[i])

            prediction_rows.append({
                "image_id": image_id,
                "actual": CLASSES[int(labels[i])],
                "predicted": CLASSES[predicted_index],
                "ensemble_score": round(
                    float(ensemble_probs[i, predicted_index]),
                    4
                )
            })


# ==========================================
# 8. EVALUATION
# ==========================================

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

report = classification_report(
    actual_labels,
    predicted_labels,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_path = os.path.join(
    RESULTS,
    "ensemble_classification_report.csv"
)

prediction_path = os.path.join(
    RESULTS,
    "ensemble_predictions.csv"
)

report_df.to_csv(report_path)
pd.DataFrame(prediction_rows).to_csv(
    prediction_path,
    index=False
)


# ==========================================
# 9. FINAL OUTPUT
# ==========================================

print("\n====================================")
print("DERMAI ENSEMBLE MODEL COMPLETED")
print("====================================")

print("Models: ResNet18 + MobileNetV3-Small + EfficientNet-B0")
print("Test images:", len(actual_labels))
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nFiles saved:")
print(report_path)
print(prediction_path)

print("\nEnsemble scores are model outputs,")
print("not calibrated medical confidence.")
