
from pathlib import Path
from collections import defaultdict, Counter
import csv
import random
import math
from PIL import Image, ImageStat

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "dataset"
IMAGE_DIR = DATA / "images"
META = DATA / "HAM10000_metadata.csv"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

# Read metadata; HAM10000 metadata may be tab-delimited
with open(META, "r", encoding="utf-8-sig", newline="") as f:
    sample = f.read(4096)
    f.seek(0)
    delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    reader = csv.DictReader(f, delimiter=delimiter)
    rows = list(reader)

if not rows or "image_id" not in rows[0] or "dx" not in rows[0]:
    raise ValueError("Metadata must contain image_id and dx columns.")

# Locate all images
image_lookup = {}
for ext in ("*.jpg", "*.jpeg", "*.png"):
    for path in IMAGE_DIR.rglob(ext):
        image_lookup[path.stem] = path

# Extract simple colour features using Pillow
def extract_features(path):
    image = Image.open(path).convert("RGB").resize((64, 64))
    features = []

    for channel in image.split():
        stats = ImageStat.Stat(channel)
        features.extend([stats.mean[0] / 255, stats.stddev[0] / 255])

        histogram = channel.histogram()
        total = sum(histogram)
        for i in range(8):
            value = sum(histogram[i * 32:(i + 1) * 32])
            features.append(value / total)

    return features

samples = []
print("Extracting image features...")

for index, row in enumerate(rows, 1):
    image_id = row["image_id"].strip()
    label = row["dx"].strip()
    path = image_lookup.get(image_id)

    if not path or not label:
        continue

    try:
        samples.append((extract_features(path), label))
    except Exception as error:
        print("Skipped:", image_id, error)

    if index % 500 == 0:
        print("Checked metadata rows:", index)

if len(samples) < 10:
    raise ValueError("Not enough images matched. Check your dataset folders.")

# Split into training and testing data
random.seed(42)
random.shuffle(samples)
split = int(len(samples) * 0.8)
train = samples[:split]
test = samples[split:]

# Build one average feature vector per disease class
class_features = defaultdict(list)

for features, label in train:
    class_features[label].append(features)

centroids = {}

for label, vectors in class_features.items():
    centroids[label] = [
        sum(vector[i] for vector in vectors) / len(vectors)
        for i in range(len(vectors[0]))
    ]

# Predict using nearest class centroid
def predict(features):
    return min(
        centroids,
        key=lambda label: sum(
            (features[i] - centroids[label][i]) ** 2
            for i in range(len(features))
        )
    )

actual = []
predicted = []

print("Evaluating baseline classifier...")

for features, label in test:
    actual.append(label)
    predicted.append(predict(features))

labels = sorted(set(actual) | set(predicted))
correct = sum(a == p for a, p in zip(actual, predicted))
accuracy = correct / len(actual)

# Calculate per-class precision, recall and F1
report_rows = []

for label in labels:
    tp = sum(a == label and p == label for a, p in zip(actual, predicted))
    fp = sum(a != label and p == label for a, p in zip(actual, predicted))
    fn = sum(a == label and p != label for a, p in zip(actual, predicted))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    report_rows.append([label, tp, fp, fn, precision, recall, f1])

# Save metrics
report_path = RESULTS / "baseline_classification_report.csv"

with open(report_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Disease_Class", "True_Positive", "False_Positive",
        "False_Negative", "Precision", "Recall", "F1_Score"
    ])
    writer.writerows(report_rows)

# Save predictions
prediction_path = RESULTS / "baseline_predictions.csv"

with open(prediction_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Actual_Class", "Predicted_Class"])
    writer.writerows(zip(actual, predicted))

print("\nBASELINE CLASSIFICATION COMPLETED")
print("Matched images:", len(samples))
print("Training images:", len(train))
print("Testing images:", len(test))
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Classes:", sorted(centroids))
print("Classification report:", report_path)
print("Predictions:", prediction_path)
