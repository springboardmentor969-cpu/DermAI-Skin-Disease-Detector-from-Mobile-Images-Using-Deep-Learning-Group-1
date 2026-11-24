from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "splits"

TRAIN_PATH = SPLITS_DIR / "train.csv"
VAL_PATH = SPLITS_DIR / "validation.csv"
TEST_PATH = SPLITS_DIR / "test.csv"

print("=" * 60)
print("DERMAI SPLIT VERIFICATION")
print("=" * 60)

print("\nLoading split files...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Training records   : {len(train_df)}")
print(f"Validation records : {len(val_df)}")
print(f"Test records       : {len(test_df)}")

# ---------------------------------------------------------
# 1. TOTAL RECORD CHECK
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TOTAL RECORD CHECK")
print("=" * 60)

total_records = len(train_df) + len(val_df) + len(test_df)

print(f"Total records across splits : {total_records}")

if total_records == 2298:
    print("PASS: All 2298 records are present.")
else:
    print("FAIL: Record count does not equal 2298.")

# ---------------------------------------------------------
# 2. DUPLICATE IMAGE CHECK
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE IMAGE CHECK")
print("=" * 60)

all_images = pd.concat(
    [
        train_df["img_id"],
        val_df["img_id"],
        test_df["img_id"]
    ],
    ignore_index=True
)

duplicate_images = all_images[all_images.duplicated()]

print(f"Duplicate images: {len(duplicate_images)}")

if len(duplicate_images) == 0:
    print("PASS: No duplicate images across splits.")
else:
    print("FAIL: Duplicate images found.")

# ---------------------------------------------------------
# 3. PATIENT OVERLAP CHECK
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PATIENT OVERLAP CHECK")
print("=" * 60)

train_patients = set(train_df["patient_id"])
val_patients = set(val_df["patient_id"])
test_patients = set(test_df["patient_id"])

train_val = train_patients.intersection(val_patients)
train_test = train_patients.intersection(test_patients)
val_test = val_patients.intersection(test_patients)

print(f"Train ∩ Validation : {len(train_val)}")
print(f"Train ∩ Test       : {len(train_test)}")
print(f"Validation ∩ Test  : {len(val_test)}")

if len(train_val) == 0 and len(train_test) == 0 and len(val_test) == 0:
    print("PASS: No patient overlap.")
else:
    print("FAIL: Patient overlap detected.")

# ---------------------------------------------------------
# 4. IMAGE PATH CHECK
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("IMAGE PATH CHECK")
print("=" * 60)

all_split_data = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

missing_paths = []

for path in all_split_data["image_path"]:
    if not Path(path).exists():
        missing_paths.append(path)

print(f"Missing image paths: {len(missing_paths)}")

if len(missing_paths) == 0:
    print("PASS: All image paths exist.")
else:
    print("FAIL: Missing image paths detected.")

    for path in missing_paths[:10]:
        print(path)

# ---------------------------------------------------------
# 5. CLASS CHECK
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CLASS CHECK")
print("=" * 60)

expected_classes = {
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK"
}

actual_classes = set(all_split_data["diagnostic"].unique())

print("Expected classes:")
print(sorted(expected_classes))

print("\nActual classes:")
print(sorted(actual_classes))

if actual_classes == expected_classes:
    print("PASS: All six disease classes are present.")
else:
    print("FAIL: Class mismatch detected.")

# ---------------------------------------------------------
# 6. CLASS DISTRIBUTION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print("\nTRAINING:")
print(train_df["diagnostic"].value_counts())

print("\nVALIDATION:")
print(val_df["diagnostic"].value_counts())

print("\nTEST:")
print(test_df["diagnostic"].value_counts())

# ---------------------------------------------------------
# FINAL STATUS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

checks_passed = (
    total_records == 2298
    and len(duplicate_images) == 0
    and len(train_val) == 0
    and len(train_test) == 0
    and len(val_test) == 0
    and len(missing_paths) == 0
    and actual_classes == expected_classes
)

if checks_passed:
    print("STATUS: ALL CHECKS PASSED")
    print("Dataset splits are ready for model training.")
else:
    print("STATUS: VERIFICATION FAILED")
    print("Please fix the failed checks before training.")