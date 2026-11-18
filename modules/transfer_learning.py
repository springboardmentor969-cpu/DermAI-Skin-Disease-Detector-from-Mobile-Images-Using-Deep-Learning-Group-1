
from pathlib import Path
from collections import Counter
import csv
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Project paths
BASE = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE / "dataset" / "images"
META = BASE / "dataset" / "HAM10000_metadata.csv"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

# Settings
SEED = 42
MAX_IMAGES = 3000       # Manageable CPU experiment
EPOCHS = 2
BATCH_SIZE = 16

random.seed(SEED)
torch.manual_seed(SEED)

print("DERMAI - RESNET18 TRANSFER LEARNING")
print("Device: CPU")

# Read metadata
with open(META, "r", encoding="utf-8-sig", newline="") as f:
    sample = f.read(4096)
    f.seek(0)
    delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    reader = csv.DictReader(f, delimiter=delimiter)
    rows = list(reader)

# Find images
image_lookup = {}
for ext in ("*.jpg", "*.jpeg", "*.png"):
    for path in IMAGE_DIR.rglob(ext):
        image_lookup[path.stem] = path

records = []
for row in rows:
    image_id = row["image_id"].strip()
    label = row["dx"].strip()
    if image_id in image_lookup and label:
        records.append((str(image_lookup[image_id]), label))

if len(records) < 100:
    raise RuntimeError("Too few matching images. Check the dataset.")

# Stratified sample for a manageable CPU demonstration
if len(records) > MAX_IMAGES:
    selected, _ = train_test_split(
        records,
        train_size=MAX_IMAGES,
        random_state=SEED,
        stratify=[r[1] for r in records]
    )
else:
    selected = records

paths = [r[0] for r in selected]
labels = [r[1] for r in selected]

classes = sorted(set(labels))
class_to_index = {name: i for i, name in enumerate(classes)}
targets = [class_to_index[label] for label in labels]

train_paths, test_paths, train_targets, test_targets = train_test_split(
    paths,
    targets,
    test_size=0.2,
    random_state=SEED,
    stratify=targets
)

# Training augmentation and validation preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.15, contrast=0.15, saturation=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class SkinDataset(Dataset):
    def __init__(self, image_paths, targets, transform):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = self.transform(image)
        return image, self.targets[index]

train_dataset = SkinDataset(
    train_paths, train_targets, train_transform
)
test_dataset = SkinDataset(
    test_paths, test_targets, test_transform
)

# Class imbalance handling with weighted sampling
counts = Counter(train_targets)
sample_weights = [
    1.0 / counts[target] for target in train_targets
]

sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Load pretrained ResNet18
print("Loading pretrained ResNet18 weights...")
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze pretrained feature layers
for parameter in model.parameters():
    parameter.requires_grad = False

# Train the new classification layer
model.fc = nn.Linear(model.fc.in_features, len(classes))

device = torch.device("cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Training
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))
print("Classes:", classes)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, targets_batch in train_loader:
        images = images.to(device)
        targets_batch = targets_batch.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets_batch).sum().item()
        total += targets_batch.size(0)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Training accuracy: {100 * correct / total:.2f}%"
    )

# Evaluation
model.eval()
actual = []
predicted = []

with torch.no_grad():
    for images, targets_batch in test_loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1)

        actual.extend(targets_batch.tolist())
        predicted.extend(predictions.cpu().tolist())

accuracy = sum(
    a == p for a, p in zip(actual, predicted)
) / len(actual)

class_names = [str(c) for c in classes]

report = classification_report(
    actual,
    predicted,
    labels=list(range(len(classes))),
    target_names=class_names,
    zero_division=0,
    output_dict=True
)

# Save classification report
report_path = RESULTS / "resnet18_classification_report.csv"

with open(report_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Precision", "Recall", "F1-score", "Support"])

    for name in class_names:
        item = report[name]
        writer.writerow([
            name,
            item["precision"],
            item["recall"],
            item["f1-score"],
            item["support"]
        ])

# Save predictions
prediction_path = RESULTS / "resnet18_predictions.csv"

with open(prediction_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Actual_Class", "Predicted_Class"])
    for a, p in zip(actual, predicted):
        writer.writerow([classes[a], classes[p]])

# Save confusion matrix
matrix = confusion_matrix(
    actual, predicted, labels=list(range(len(classes)))
)

with open(RESULTS / "resnet18_confusion_matrix.csv",
          "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Class"] + class_names)
    for name, row in zip(class_names, matrix):
        writer.writerow([name] + row.tolist())

# Save trained model
model_path = RESULTS / "dermai_resnet18.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes,
    "accuracy": accuracy
}, model_path)

print("\nTRANSFER LEARNING COMPLETED")
print("Model: ResNet18 pretrained")
print("Images used:", len(selected))
print("Test accuracy:", round(accuracy * 100, 2), "%")
print("Model saved:", model_path)
print("Classification report:", report_path)
print("Predictions:", prediction_path)
print("Confusion matrix saved.")
