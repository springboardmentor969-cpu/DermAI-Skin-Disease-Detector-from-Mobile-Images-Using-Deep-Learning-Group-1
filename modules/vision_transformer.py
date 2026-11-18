
from pathlib import Path
import csv
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE / "dataset" / "images"
META = BASE / "dataset" / "HAM10000_metadata.csv"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
LIMIT = 1200
EPOCHS = 1
BATCH_SIZE = 8

random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")

print("DERMAI - VISION TRANSFORMER")
print("Device: CPU")

# Read metadata
with open(META, encoding="utf-8-sig", newline="") as f:
    sample = f.read(4096)
    f.seek(0)
    delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    rows = list(csv.DictReader(f, delimiter=delimiter))

# Locate images
lookup = {}
for ext in ("*.jpg", "*.jpeg", "*.png"):
    for path in IMAGE_DIR.rglob(ext):
        lookup[path.stem] = path

records = [
    (str(lookup[r["image_id"].strip()]), r["dx"].strip())
    for r in rows
    if r["image_id"].strip() in lookup and r["dx"].strip()
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

# Image preprocessing
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

# Handle class imbalance
counts = Counter(train_y)
weights = [1.0 / counts[y] for y in train_y]

sampler = WeightedRandomSampler(
    weights,
    num_samples=len(weights),
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


# Vision Transformer architecture
class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        image_size=224,
        patch_size=16,
        embed_dim=192,
        num_heads=3,
        num_layers=4
    ):
        super().__init__()

        self.patch_embedding = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        number_of_patches = (image_size // patch_size) ** 2

        self.class_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, number_of_patches + 1, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        batch_size = x.shape[0]

        class_token = self.class_token.expand(
            batch_size, -1, -1
        )

        x = torch.cat([class_token, x], dim=1)
        x = x + self.position_embedding

        x = self.transformer(x)
        x = self.norm(x[:, 0])

        return self.classifier(x)


model = VisionTransformer(
    num_classes=len(classes)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.01
)

print("Training images:", len(train_ds))
print("Testing images:", len(test_ds))
print("Classes:", classes)
print("Architecture: Vision Transformer")
print("Patch size: 16 x 16")
print("Transformer layers: 4")

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

# Save report
report_path = RESULTS / "vision_transformer_report.csv"

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
prediction_path = RESULTS / "vision_transformer_predictions.csv"

with open(prediction_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Actual_Class", "Predicted_Class"])

    for a, p in zip(actual, predicted):
        writer.writerow([classes[a], classes[p]])

# Save model
model_path = RESULTS / "dermai_vision_transformer.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes,
    "accuracy": accuracy,
    "architecture": "Vision Transformer",
    "patch_size": 16,
    "embed_dim": 192,
    "num_heads": 3,
    "num_layers": 4
}, model_path)

print("\nVISION TRANSFORMER COMPLETED")
print("Test accuracy:", round(accuracy * 100, 2), "%")
print("Model saved:", model_path)
print("Report saved:", report_path)
print("Predictions saved:", prediction_path)
