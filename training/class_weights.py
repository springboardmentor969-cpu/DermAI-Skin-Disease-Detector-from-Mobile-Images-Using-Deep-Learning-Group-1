import torch
import pandas as pd

from pathlib import Path


# ---------------------------------------------------------
# PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = PROJECT_ROOT / "splits" / "train.csv"


# ---------------------------------------------------------
# CLASS ORDER
# ---------------------------------------------------------

CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK"
]


# ---------------------------------------------------------
# CALCULATE CLASS WEIGHTS
# ---------------------------------------------------------

def calculate_class_weights():

    df = pd.read_csv(TRAIN_CSV)

    class_counts = (
        df["diagnostic"]
        .value_counts()
        .reindex(CLASS_NAMES)
    )

    total_samples = len(df)
    num_classes = len(CLASS_NAMES)

    weights = total_samples / (
        num_classes * class_counts
    )

    weights = torch.tensor(
        weights.values,
        dtype=torch.float32
    )

    return class_counts, weights


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    class_counts, weights = calculate_class_weights()

    print("=" * 60)
    print("DERMAI CLASS WEIGHTS")
    print("=" * 60)

    print("\nClass counts:")

    for class_name, count in zip(
        CLASS_NAMES,
        class_counts
    ):
        print(f"{class_name}: {count}")

    print("\nClass weights:")

    for class_name, weight in zip(
        CLASS_NAMES,
        weights
    ):
        print(f"{class_name}: {weight.item():.4f}")

    print("\n" + "=" * 60)
    print("CLASS WEIGHT CALCULATION COMPLETE")
    print("=" * 60)