
from pathlib import Path
import csv
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import (
    ResNet18_Weights,
    MobileNet_V3_Small_Weights
)
from PIL import Image
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
BATCH = 16

random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")

# Read metadata
with open(META, encoding="utf-8-sig", newline="") as f:
    sample = f.read(4096)
    f.seek(0)
    delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    rows = list(csv.DictReader(f, delimiter=delimiter))

# Locate images
lookup = {}
for ext in ("*.jpg", "*.jpeg", "*.png"):
    for p in IMAGE_DIR.rglob(ext):
        lookup[p.stem] = p

records = [
    (str(lookup[r["image_id"].strip()]), r["dx"].strip())
    for r in rows
    if r["image_id"].strip() in lookup and r["dx"].strip()
]

# Stratified sample
if len(records) > LIMIT:
    records, _ = train_test_split(
        records,
        train_size=LIMIT,
        random_state=SEED,
        stratify=[r[1] for r in records]
    )

paths = [r[0] for r in records]
names = sorted(set(r[1] for r in records))
mapping = {name: i for i, name in enumerate(names)}
targets = [mapping[r[1]] for r in records]

train_p, test_p, train_y, test_y = train_test_split(
    paths, targets, test_size=0.2,
    random_state=SEED, stratify=targets
)

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

class SkinData(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.transform(image), self.labels[i]

train_ds = SkinData(train_p, train_y, train_tf)
test_ds = SkinData(test_p, test_y, test_tf)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

def train_and_evaluate(model, model_name):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0005
    )

    print("\nTraining:", model_name)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in train_dl:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            "Epoch", epoch + 1,
            "Loss:", round(total_loss / len(train_dl), 4)
        )

    model.eval()
    actual, predicted = [], []

    with torch.no_grad():
        for images, labels in test_dl:
            output = model(images.to(device))
            predicted.extend(output.argmax(1).tolist())
            actual.extend(labels.tolist())

    accuracy = accuracy_score(actual, predicted)

    report = classification_report(
        actual, predicted,
        labels=list(range(len(names))),
        target_names=names,
        zero_division=0,
        output_dict=True
    )

    with open(
        RESULTS / f"{model_name}_report.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Precision", "Recall", "F1", "Support"])
        for name in names:
            item = report[name]
            writer.writerow([
                name, item["precision"], item["recall"],
                item["f1-score"], item["support"]
            ])

    torch.save(
        {"model_state_dict": model.state_dict(), "classes": names},
        RESULTS / f"{model_name}.pth"
    )

    print(model_name, "accuracy:", round(accuracy * 100, 2), "%")
    return accuracy

# Model 1: ResNet18
resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for p in resnet.parameters():
    p.requires_grad = False
resnet.fc = nn.Linear(resnet.fc.in_features, len(names))
resnet_accuracy = train_and_evaluate(resnet, "resnet18_comparison")

# Model 2: MobileNetV3 Small
mobile = models.mobilenet_v3_small(
    weights=MobileNet_V3_Small_Weights.DEFAULT
)
for p in mobile.parameters():
    p.requires_grad = False
mobile.classifier[3] = nn.Linear(
    mobile.classifier[3].in_features, len(names)
)
mobile_accuracy = train_and_evaluate(
    mobile, "mobilenet_v3_small"
)

# Save comparison
comparison = [
    ["ResNet18", resnet_accuracy],
    ["MobileNetV3-Small", mobile_accuracy]
]

with open(
    RESULTS / "cnn_model_comparison.csv",
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Test_Accuracy"])
    writer.writerows(comparison)

print("\nCNN COMPARISON COMPLETED")
print("ResNet18:", round(resnet_accuracy * 100, 2), "%")
print("MobileNetV3-Small:", round(mobile_accuracy * 100, 2), "%")
print("Comparison saved in results/cnn_model_comparison.csv")
