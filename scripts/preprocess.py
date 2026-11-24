from pathlib import Path
import sys

import numpy as np
from PIL import Image

# Allow importing from the DermAI project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.image_preprocessing import preprocess_image


# ============================================================
# PATHS
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\SRINJOY\Dropbox\PC\Downloads\PAD-UFES-20"
)

IMAGES_DIR = DATASET_ROOT / "images"

PROCESSED_DIR = PROJECT_ROOT / "processed_images"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("PAD-UFES-20 IMAGE PREPROCESSING")
print("=" * 60)

print("\nOriginal images:")
print(IMAGES_DIR)

print("\nProcessed images:")
print(PROCESSED_DIR)

print("\nSearching for images...")

image_paths = list(IMAGES_DIR.rglob("*.png"))

print(f"Images found: {len(image_paths)}")

if len(image_paths) == 0:
    print("\nERROR: No PNG images found.")
    sys.exit(1)


processed_count = 0
failed_count = 0


# ============================================================
# PROCESS IMAGES
# ============================================================

for index, image_path in enumerate(image_paths, start=1):

    try:

        # Preprocess image
        image_array = preprocess_image(image_path)

        # Convert normalized [0,1] values back to [0,255]
        # for saving as PNG.
        output_array = (
            image_array * 255
        ).clip(0, 255).astype(np.uint8)

        # Preserve original filename
        output_path = PROCESSED_DIR / image_path.name

        # Save processed image
        output_image = Image.fromarray(
            output_array
        )

        output_image.save(output_path)

        processed_count += 1

        if index % 100 == 0 or index == len(image_paths):

            print(
                f"Processed {index}/{len(image_paths)}"
            )

    except Exception as error:

        failed_count += 1

        print(
            f"ERROR processing {image_path.name}: {error}"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"Total images found : {len(image_paths)}")
print(f"Successfully processed : {processed_count}")
print(f"Failed : {failed_count}")

print("\nProcessed dataset:")
print(PROCESSED_DIR)

print("=" * 60)