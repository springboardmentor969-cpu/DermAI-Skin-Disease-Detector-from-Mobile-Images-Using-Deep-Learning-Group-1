
import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
BATCH_SIZE = 16
EPOCHS = 1

LEARNING_RATES = [0.001, 0.0001, 0.00001]

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("DERMAI HYPERPARAMETER OPTIMIZATION STARTED")


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

sample_size = min(LIMIT, len(df))

df = df.sample(
    n=sample_size,
    random_state=SEED
).reset_index(drop=True)

if df["dx"].value_counts().min() < 2:
    raise ValueError("Not enough samples for stratified splitting.")

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["dx"]
)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))


# ==========================================
# 4. IMAGE TRANSFORMS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# 5. DATASET CLASS
# ==========================================

class HAMDataset(Dataset):

    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

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

        return image, label


train_dataset = HAMDataset(train_df, train_transform)
val_dataset = HAMDataset(val_df, val_transform)

# Balance training samples across classes
class_counts = train_df["dx"].value_counts()

sample_weights = [
    1.0 / class_counts[label]
    for label in train_df["dx"]
]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 6. HYPERPARAMETER EXPERIMENTS
# ==========================================

results = []

best_accuracy = -1
best_learning_rate = None
best_state = None

for learning_rate in LEARNING_RATES:

    print("\n--------------------------------")
    print("Testing learning rate:", learning_rate)
    print("--------------------------------")

    torch.manual_seed(SEED)

    # Start every trial from the same pretrained weights
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(CLASSES)
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # Training
    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0.0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / max(1, len(train_loader))

        print(
            "Epoch",
            epoch + 1,
            "Loss:",
            round(average_loss, 4)
        )

    # Validation
    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            actual.extend(labels.tolist())
            predicted.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(actual, predicted)

    print("Validation accuracy:", round(accuracy * 100, 2), "%")

    results.append({
        "Model": "EfficientNet-B0",
        "Learning Rate": learning_rate,
        "Epochs": EPOCHS,
        "Training Images": len(train_df),
        "Validation Images": len(val_df),
        "Validation Accuracy": round(accuracy * 100, 2),
        "Training Loss": round(average_loss, 4)
    })

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_learning_rate = learning_rate

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }


# ==========================================
# 7. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

report_path = os.path.join(
    RESULTS,
    "hyperparameter_optimization.csv"
)

results_df.to_csv(report_path, index=False)

if best_state is not None:

    model_path = os.path.join(
        RESULTS,
        "dermai_hpo_efficientnet_b0.pth"
    )

    torch.save({
        "model_state_dict": best_state,
        "classes": CLASSES,
        "best_learning_rate": best_learning_rate,
        "validation_accuracy": best_accuracy
    }, model_path)

else:
    model_path = "No model saved"


# ==========================================
# 8. FINAL SUMMARY
# ==========================================

print("\n====================================")
print("HYPERPARAMETER OPTIMIZATION COMPLETED")
print("====================================")

print("\nExperiment results:")
print(results_df.to_string(index=False))

print("\nBest learning rate:", best_learning_rate)
print("Best validation accuracy:", round(best_accuracy * 100, 2), "%")

print("\nReport saved:", report_path)
print("Best model saved:", model_path)
