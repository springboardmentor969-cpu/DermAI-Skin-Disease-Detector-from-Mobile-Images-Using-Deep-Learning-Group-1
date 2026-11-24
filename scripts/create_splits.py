from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = Path(
    r"C:\Users\SRINJOY\Dropbox\PC\Downloads\PAD-UFES-20"
)

METADATA_PATH = DATASET_ROOT / "metadata.csv"

SPLITS_DIR = PROJECT_ROOT / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 60)
print("PAD-UFES-20 PATIENT-LEVEL DATASET SPLITTING")
print("=" * 60)

print("\nLoading metadata...")

df = pd.read_csv(METADATA_PATH)

print(f"Total records: {len(df)}")
print(f"Unique patients: {df['patient_id'].nunique()}")


# ============================================================
# CREATE PATIENT-LEVEL TABLE
# ============================================================

# Each patient must belong to exactly one split.
#
# We need one disease label per patient for stratification.
# If a patient has multiple diagnostic classes, we assign the
# most frequent class for the purpose of stratifying patients.

patient_labels = (
    df.groupby("patient_id")["diagnostic"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
)

print(
    f"Patients available for splitting: "
    f"{len(patient_labels)}"
)


# ============================================================
# FIRST SPLIT
# TRAIN = 70%
# TEMP  = 30%
# ============================================================

train_patients, temp_patients = train_test_split(
    patient_labels,
    test_size=(VALIDATION_SIZE + TEST_SIZE),
    random_state=RANDOM_STATE,
    stratify=patient_labels["diagnostic"]
)


# ============================================================
# SECOND SPLIT
# VALIDATION = 15%
# TEST       = 15%
#
# TEMP is 30%, so each becomes 50% of TEMP.
# ============================================================

validation_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_patients["diagnostic"]
)


# ============================================================
# GET PATIENT IDS
# ============================================================

train_patient_ids = set(
    train_patients["patient_id"]
)

validation_patient_ids = set(
    validation_patients["patient_id"]
)

test_patient_ids = set(
    test_patients["patient_id"]
)


# ============================================================
# SAFETY CHECK
# ============================================================

train_validation_overlap = (
    train_patient_ids &
    validation_patient_ids
)

train_test_overlap = (
    train_patient_ids &
    test_patient_ids
)

validation_test_overlap = (
    validation_patient_ids &
    test_patient_ids
)


print("\n" + "=" * 60)
print("PATIENT OVERLAP CHECK")
print("=" * 60)

print(
    "Train ∩ Validation:",
    len(train_validation_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(validation_test_overlap)
)


if (
    len(train_validation_overlap) != 0
    or len(train_test_overlap) != 0
    or len(validation_test_overlap) != 0
):

    raise RuntimeError(
        "ERROR: Patient leakage detected!"
    )


# ============================================================
# ASSIGN SPLIT TO EVERY IMAGE RECORD
# ============================================================

def assign_split(patient_id):

    if patient_id in train_patient_ids:
        return "train"

    if patient_id in validation_patient_ids:
        return "validation"

    if patient_id in test_patient_ids:
        return "test"

    raise ValueError(
        f"Unknown patient ID: {patient_id}"
    )


df["split"] = df["patient_id"].apply(assign_split)


# ============================================================
# ADD PROCESSED IMAGE PATH
# ============================================================

PROCESSED_DIR = PROJECT_ROOT / "processed_images"

df["image_path"] = df["img_id"].apply(
    lambda x: str(PROCESSED_DIR / x)
)


# ============================================================
# SAVE SPLITS
# ============================================================

train_df = df[df["split"] == "train"].copy()
validation_df = df[
    df["split"] == "validation"
].copy()
test_df = df[df["split"] == "test"].copy()


train_path = SPLITS_DIR / "train.csv"
validation_path = SPLITS_DIR / "validation.csv"
test_path = SPLITS_DIR / "test.csv"


train_df.to_csv(
    train_path,
    index=False
)

validation_df.to_csv(
    validation_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SPLIT RESULTS")
print("=" * 60)

print(
    f"Training patients   : {len(train_patient_ids)}"
)

print(
    f"Validation patients : {len(validation_patient_ids)}"
)

print(
    f"Test patients       : {len(test_patient_ids)}"
)

print()

print(
    f"Training images     : {len(train_df)}"
)

print(
    f"Validation images   : {len(validation_df)}"
)

print(
    f"Test images         : {len(test_df)}"
)

print()

print(
    f"Training percentage : "
    f"{len(train_df) / len(df) * 100:.2f}%"
)

print(
    f"Validation percentage : "
    f"{len(validation_df) / len(df) * 100:.2f}%"
)

print(
    f"Test percentage : "
    f"{len(test_df) / len(df) * 100:.2f}%"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION BY SPLIT")
print("=" * 60)

print("\nTRAINING:")
print(train_df["diagnostic"].value_counts())

print("\nVALIDATION:")
print(validation_df["diagnostic"].value_counts())

print("\nTEST:")
print(test_df["diagnostic"].value_counts())


# ============================================================
# FILE LOCATIONS
# ============================================================

print("\n" + "=" * 60)
print("SPLIT FILES CREATED")
print("=" * 60)

print(train_path)
print(validation_path)
print(test_path)

print("\n" + "=" * 60)
print("PATIENT-LEVEL SPLITTING COMPLETE")
print("=" * 60)