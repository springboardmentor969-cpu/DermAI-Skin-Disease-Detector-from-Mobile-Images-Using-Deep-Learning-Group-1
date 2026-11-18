
import os
import csv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")

files = {
    "Traditional Baseline": "baseline_classification_report.csv",
    "ResNet18 Transfer Learning": "resnet18_classification_report.csv",
    "ResNet18 CNN": "resnet18_comparison_report.csv",
    "MobileNetV3 CNN": "mobilenet_v3_small_report.csv"
}

summary = []

for model, filename in files.items():
    path = os.path.join(RESULTS, filename)

    if not os.path.exists(path):
        print("Missing:", filename)
        continue

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    weighted_total = 0.0
    total_support = 0.0
    correct_accuracy = None

    for row in rows:
        label = (row.get("Class") or "").strip()

        if label == "accuracy":
            correct_accuracy = float(row["F1"])
        elif label in ["macro avg", "weighted avg"]:
            continue
        else:
            try:
                f1 = float(row["F1"])
                support = float(row["Support"])
                weighted_total += f1 * support
                total_support += support
            except (ValueError, TypeError, KeyError):
                continue

    if total_support == 0:
        print("Could not calculate score:", filename)
        continue

    weighted_f1 = weighted_total / total_support

    summary.append([
        model,
        correct_accuracy,
        weighted_f1
    ])

if summary:
    output = os.path.join(RESULTS, "final_model_comparison.csv")

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "Weighted F1"])
        writer.writerows(summary)

    print("\nFINAL MODEL COMPARISON")
    print("=" * 60)

    for model, accuracy, f1 in summary:
        print(model)
        print("Accuracy:", accuracy if accuracy is not None else "Not available")
        print("Weighted F1:", round(f1, 4))
        print("-" * 40)

    best = max(summary, key=lambda x: x[2])

    print("\nBEST MODEL:", best[0])
    print("Weighted F1:", round(best[2], 4))
    print("\nSaved:", output)

else:
    print("No valid reports found.")
