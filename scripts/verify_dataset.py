from pathlib import Path
import sys
import pandas as pd
from PIL import Image


# ============================================================
# 1. PAD-UFES-20 DATASET LOCATION
# ============================================================

PAD_UFES_ROOT = Path(r"C:\Users\SRINJOY\Dropbox\PC\Downloads\PAD-UFES-20")

METADATA_PATH = PAD_UFES_ROOT / "metadata.csv"
IMAGES_PATH = PAD_UFES_ROOT / "images"


# ============================================================
# 2. CHECK DATASET PATHS
# ============================================================

print("=" * 60)
print("PAD-UFES-20 DATASET VERIFICATION")
print("=" * 60)

print("\nDataset root:")
print(PAD_UFES_ROOT)

print("\nMetadata:")
print(METADATA_PATH)

print("\nImages:")
print(IMAGES_PATH)


if not PAD_UFES_ROOT.exists():
    print("\nERROR: Dataset root folder does not exist.")
    print("Check PAD_UFES_ROOT in this script.")
    sys.exit(1)

if not METADATA_PATH.exists():
    print("\nERROR: metadata.csv was not found.")
    sys.exit(1)

if not IMAGES_PATH.exists():
    print("\nERROR: images folder was not found.")
    sys.exit(1)


# ============================================================
# 3. LOAD METADATA
# ============================================================

print("\nLoading metadata.csv...")

df = pd.read_csv(METADATA_PATH)

print("Metadata loaded successfully.")

print("\nNumber of metadata records:", len(df))
print("Number of metadata columns:", len(df.columns))


# ============================================================
# 4. FIND ALL IMAGE FILES
# ============================================================

print("\nScanning image folders...")
print("This may take a little time.")

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_files = []

for path in IMAGES_PATH.rglob("*"):
    if path.is_file() and path.suffix.lower() in image_extensions:
        image_files.append(path)


print("\nTotal image files found:", len(image_files))


# ============================================================
# 5. COUNT IMAGES BY FOLDER
# ============================================================

print("\n" + "=" * 60)
print("IMAGE COUNT BY TOP-LEVEL FOLDER")
print("=" * 60)

folder_counts = {}

for image_path in image_files:

    relative_path = image_path.relative_to(IMAGES_PATH)

    if len(relative_path.parts) > 1:
        top_folder = relative_path.parts[0]
    else:
        top_folder = "(images directly inside images folder)"

    folder_counts[top_folder] = folder_counts.get(top_folder, 0) + 1


for folder, count in sorted(folder_counts.items()):
    print(f"{folder}: {count}")


# ============================================================
# 6. CHECK IMAGE IDS AGAINST METADATA
# ============================================================

print("\n" + "=" * 60)
print("METADATA ↔ IMAGE MATCHING")
print("=" * 60)

metadata_image_ids = set(
    df["img_id"]
    .dropna()
    .astype(str)
)

actual_image_names = set(
    image_path.name
    for image_path in image_files
)


missing_images = metadata_image_ids - actual_image_names
extra_images = actual_image_names - metadata_image_ids


print("Images expected from metadata:", len(metadata_image_ids))
print("Images actually found:", len(actual_image_names))
print("Metadata images missing from folders:", len(missing_images))
print("Images without metadata records:", len(extra_images))


if missing_images:
    print("\nFirst 20 missing images:")

    for name in sorted(missing_images)[:20]:
        print(name)


if extra_images:
    print("\nFirst 20 extra images:")

    for name in sorted(extra_images)[:20]:
        print(name)


# ============================================================
# 7. CHECK IMAGE READABILITY
# ============================================================

print("\n" + "=" * 60)
print("IMAGE READABILITY CHECK")
print("=" * 60)

corrupted_images = []

checked = 0

for image_path in image_files:

    try:
        with Image.open(image_path) as img:
            img.verify()

        checked += 1

    except Exception:
        corrupted_images.append(image_path)


print("Images successfully checked:", checked)
print("Corrupted/unreadable images:", len(corrupted_images))


if corrupted_images:

    print("\nCorrupted images:")

    for image_path in corrupted_images[:20]:
        print(image_path)


# ============================================================
# 8. IMAGE FORMAT DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("IMAGE FORMAT DISTRIBUTION")
print("=" * 60)

format_counts = {}

for image_path in image_files:

    extension = image_path.suffix.lower()

    format_counts[extension] = (
        format_counts.get(extension, 0) + 1
    )


for extension, count in sorted(format_counts.items()):
    print(f"{extension}: {count}")


# ============================================================
# 9. IMAGE DIMENSIONS
# ============================================================

print("\n" + "=" * 60)
print("IMAGE DIMENSION ANALYSIS")
print("=" * 60)

dimension_counts = {}
dimension_errors = []

for image_path in image_files:

    try:

        with Image.open(image_path) as img:

            width, height = img.size

            dimension = (width, height)

            dimension_counts[dimension] = (
                dimension_counts.get(dimension, 0) + 1
            )

    except Exception:

        dimension_errors.append(image_path)


print("Unique image dimensions:", len(dimension_counts))

print("\nMost common dimensions:")

sorted_dimensions = sorted(
    dimension_counts.items(),
    key=lambda x: x[1],
    reverse=True
)

for dimension, count in sorted_dimensions[:20]:

    print(
        f"{dimension[0]} x {dimension[1]}: "
        f"{count} images"
    )


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET VERIFICATION SUMMARY")
print("=" * 60)

print(f"""
Metadata records             : {len(df)}
Metadata columns             : {len(df.columns)}
Image files found            : {len(image_files)}
Expected image IDs           : {len(metadata_image_ids)}
Missing images               : {len(missing_images)}
Extra images                 : {len(extra_images)}
Corrupted/unreadable images  : {len(corrupted_images)}
Unique image dimensions      : {len(dimension_counts)}
""")

print("=" * 60)
print("DATASET VERIFICATION COMPLETE")
print("=" * 60)