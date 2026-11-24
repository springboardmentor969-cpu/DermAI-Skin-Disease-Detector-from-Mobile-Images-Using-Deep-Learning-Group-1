from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PAD-UFES-20 DATASET EDA
# ============================================================

# Dataset location
DATASET_ROOT = Path(
    r"C:\Users\SRINJOY\Dropbox\PC\Downloads\PAD-UFES-20"
)

METADATA_PATH = DATASET_ROOT / "metadata.csv"

# Output folder inside DermAI project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 60)
print("PAD-UFES-20 EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nLoading metadata.csv...")

df = pd.read_csv(METADATA_PATH)

print("Metadata loaded successfully.")

print("\n" + "=" * 60)
print("BASIC DATASET INFORMATION")
print("=" * 60)

print(f"Number of records : {len(df)}")
print(f"Number of columns : {len(df.columns)}")

print("\nColumns:")
for column in df.columns:
    print(f"- {column}")


# ============================================================
# 1. DISEASE / DIAGNOSTIC DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("DISEASE / DIAGNOSTIC DISTRIBUTION")
print("=" * 60)

diagnostic_counts = df["diagnostic"].value_counts()

print(diagnostic_counts)

print("\nPercentages:")
print((df["diagnostic"].value_counts(normalize=True) * 100).round(2))


plt.figure(figsize=(9, 6))
diagnostic_counts.plot(kind="bar")
plt.title("PAD-UFES-20 Disease Class Distribution")
plt.xlabel("Disease Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300)
plt.close()


# ============================================================
# 2. AGE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("AGE ANALYSIS")
print("=" * 60)

print(df["age"].describe())

plt.figure(figsize=(9, 6))
df["age"].dropna().plot(kind="hist", bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients/Records")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "age_distribution.png", dpi=300)
plt.close()


# ============================================================
# 3. GENDER DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("GENDER DISTRIBUTION")
print("=" * 60)

gender_counts = df["gender"].fillna("Missing").value_counts()

print(gender_counts)

plt.figure(figsize=(8, 6))
gender_counts.plot(kind="bar")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "gender_distribution.png", dpi=300)
plt.close()


# ============================================================
# 4. BODY REGION DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("BODY REGION DISTRIBUTION")
print("=" * 60)

region_counts = df["region"].value_counts()

print(region_counts)

plt.figure(figsize=(10, 7))
region_counts.plot(kind="bar")
plt.title("Lesion Location / Body Region Distribution")
plt.xlabel("Body Region")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "region_distribution.png", dpi=300)
plt.close()


# ============================================================
# 5. FITZPATRICK SKIN TYPE
# ============================================================

print("\n" + "=" * 60)
print("FITZPATRICK SKIN TYPE DISTRIBUTION")
print("=" * 60)

fitz_counts = df["fitspatrick"].fillna("Missing").value_counts()

print(fitz_counts)

plt.figure(figsize=(9, 6))
fitz_counts.plot(kind="bar")
plt.title("Fitzpatrick Skin Type Distribution")
plt.xlabel("Fitzpatrick Type")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "fitzpatrick_distribution.png", dpi=300)
plt.close()


# ============================================================
# 6. BIOPSY DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("BIOPSY DISTRIBUTION")
print("=" * 60)

biopsy_counts = df["biopsed"].value_counts()

print(biopsy_counts)

plt.figure(figsize=(8, 6))
biopsy_counts.plot(kind="bar")
plt.title("Biopsy Status Distribution")
plt.xlabel("Biopsied")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "biopsy_distribution.png", dpi=300)
plt.close()


# ============================================================
# 7. SYMPTOM / CLINICAL FEATURE DISTRIBUTION
# ============================================================

clinical_features = [
    "itch",
    "grew",
    "hurt",
    "changed",
    "bleed",
    "elevation"
]

print("\n" + "=" * 60)
print("CLINICAL FEATURE DISTRIBUTION")
print("=" * 60)

for feature in clinical_features:

    print(f"\n{feature.upper()}")

    counts = df[feature].value_counts(dropna=False)

    print(counts)


# ============================================================
# 8. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_counts = df.isnull().sum()

missing_counts = missing_counts[
    missing_counts > 0
].sort_values(ascending=False)

print(missing_counts)

plt.figure(figsize=(10, 7))
missing_counts.plot(kind="bar")
plt.title("Missing Values by Column")
plt.xlabel("Column")
plt.ylabel("Number of Missing Values")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "missing_values.png", dpi=300)
plt.close()


# ============================================================
# 9. UNIQUE PATIENT / LESION / IMAGE COUNTS
# ============================================================

print("\n" + "=" * 60)
print("UNIQUE ENTITY COUNTS")
print("=" * 60)

print(f"Unique patients : {df['patient_id'].nunique()}")
print(f"Unique lesions  : {df['lesion_id'].nunique()}")
print(f"Unique images   : {df['img_id'].nunique()}")


# ============================================================
# 10. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)

print(f"""
Total records          : {len(df)}
Total columns          : {len(df.columns)}
Unique patients        : {df['patient_id'].nunique()}
Unique lesions         : {df['lesion_id'].nunique()}
Unique images          : {df['img_id'].nunique()}

Disease classes        : {df['diagnostic'].nunique()}
Disease class names    : {', '.join(df['diagnostic'].dropna().unique())}

Minimum age            : {df['age'].min()}
Maximum age            : {df['age'].max()}
Average age            : {df['age'].mean():.2f}
Median age             : {df['age'].median():.2f}
""")

print("\nEDA charts saved to:")
print(OUTPUT_DIR)

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)