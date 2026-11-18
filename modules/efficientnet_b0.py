
from pathlib import Path
from collections import Counter
import csv
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Project paths
BASE = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE / "dataset" / "images"
META = BASE / "dataset" / "HAM10000_metadata.csv"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
LIMIT = 1200
EPOCHS = 1
BATCH_SIZE = 16

random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")

print("DERMAI - EFFICIENTNET-B0 ADVANCED CNN")
print("Device: CPU")

# Read metadata
with open(META, encoding="utf-8-sig", newline="") as f:
    sample = f.read(4096)
    f.seek(0)
    delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    rows = list(csv.DictReader(f, delimiter=delimiter))

# Locate dataset images
lookup = {}
for ext in ("*.jpg", "*.jpeg", "*.png"):
    for path in IMAGE_DIR.rglob(ext):
        lookup[path.stem] = path

records = [
    (str(lookup[row["image_id"].strip()]), row["dx"].strip())
    for row in rows
    if row["image_id"].strip() in lookup and row["dx"].strip()
]

if len(records) < 100:
    raise RuntimeError("Not enough matching dataset images.")

# Stratified sample
if len(records) > LIMIT:
    records, _ = train_test_split(
        records,
        train_size=LIMIT,
        random_state=SEED,
        stratify=[r[1] for r in records]
    )

paths = [r[0] for r in records]
classes = sorted(set(r[1] for r in records))
class_map = {name: i for i, name in enumerate(classes)}
targets = [class_map[r[1]] for r in records]

train_paths, test_paths, train_y, test_y = train_test_split(
    paths,
    targets,
    test_size=0.2,
    random_state=SEED,
    stratify=targets
)

# Image transformations
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

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

class SkinDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image), self.labels[index]

train_ds = SkinDataset(train_paths, train_y, train_transform)
test_ds = SkinDataset(test_paths, test_y, test_transform)

# Weighted sampling for class imbalance
counts = Counter(train_y)
sample_weights = [1.0 / counts[y] for y in train_y]

sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Load pretrained EfficientNet-B0
print("Loading pretrained EfficientNet-B0...")
model = models.efficientnet_b0(
    weights=EfficientNet_B0_Weights.DEFAULT
)

# Freeze feature extractor
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace final classifier
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.classifier[1].parameters(),
    lr=0.001
)

print("Training images:", len(train_ds))
print("Testing images:", len(test_ds))
print("Classes:", classes)

# Training
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )

# Evaluation
model.eval()
actual = []
predicted = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        predicted.extend(outputs.argmax(1).tolist())
        actual.extend(labels.tolist())

accuracy = accuracy_score(actual, predicted)

report = classification_report(
    actual,
    predicted,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0,
    output_dict=True
)

# Save classification report
report_path = RESULTS / "efficientnet_b0_report.csv"

with open(report_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Precision", "Recall", "F1-score", "Support"])

    for name in classes:
        item = report[name]
        writer.writerow([
            name,
            item["precision"],
            item["recall"],
            item["f1-score"],
            item["support"]
        ])

# Save predictions
prediction_path = RESULTS / "efficientnet_b0_predictions.csv"

with open(prediction_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Actual_Class", "Predicted_Class"])

    for a, p in zip(actual, predicted):
        writer.writerow([classes[a], classes[p]])

# Save model
model_path = RESULTS / "dermai_efficientnet_b0.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes,
    "accuracy": accuracy,
    "architecture": "EfficientNet-B0"
}, model_path)

print("\nEFFICIENTNET-B0 COMPLETED")
print("Images used:", len(records))
print("Test accuracy:", round(accuracy * 100, 2), "%")
print("Model saved:", model_path)
print("Report saved:", report_path)
print("Predictions saved:", prediction_path)
