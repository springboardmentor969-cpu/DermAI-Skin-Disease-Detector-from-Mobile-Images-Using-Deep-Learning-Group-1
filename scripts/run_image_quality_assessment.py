from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORTS
# ============================================================

from dataset.dataset_config import IMAGES_PATH

from preprocessing.image_quality import (
    assess_image_quality
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "image_quality_report.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DERMAI IMAGE QUALITY ASSESSMENT")
    print("=" * 60)

    print()
    print("Image root:")
    print(IMAGES_PATH)

    if not IMAGES_PATH.exists():
        print()
        print("ERROR: Image directory does not exist.")
        print(IMAGES_PATH)
        return

    # Search recursively for image files.
    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    ]:
        image_files.extend(
            IMAGES_PATH.rglob(extension)
        )

    # Remove duplicates
    image_files = sorted(
        set(image_files)
    )

    print()
    print(
        f"Images found: {len(image_files)}"
    )

    if not image_files:
        print()
        print("ERROR: No images found.")
        return

    print()
    print("Running quality assessment...")
    print()

    results = []

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        result = assess_image_quality(
            image_path
        )

        results.append(result)

        if index % 100 == 0:
            print(
                f"Processed: {index}/{len(image_files)}"
            )

    # Create dataframe
    dataframe = pd.DataFrame(
        results
    )

    # Save report
    dataframe.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(dataframe)

    good = int(
        (dataframe["quality_status"] == "GOOD").sum()
    )

    warning = int(
        (dataframe["quality_status"] == "WARNING").sum()
    )

    poor = int(
        (dataframe["quality_status"] == "POOR").sum()
    )

    invalid = int(
        (dataframe["quality_status"] == "INVALID").sum()
    )

    print()
    print("=" * 60)
    print("IMAGE QUALITY SUMMARY")
    print("=" * 60)

    print(
        f"Total images : {total}"
    )

    print(
        f"GOOD         : {good}"
    )

    print(
        f"WARNING      : {warning}"
    )

    print(
        f"POOR         : {poor}"
    )

    print(
        f"INVALID      : {invalid}"
    )

    print()
    print("Average metrics:")

    print(
        f"Blur score   : "
        f"{dataframe['blur_score'].mean():.2f}"
    )

    print(
        f"Brightness   : "
        f"{dataframe['brightness'].mean():.2f}"
    )

    print(
        f"Contrast     : "
        f"{dataframe['contrast'].mean():.2f}"
    )

    print()
    print("Report saved to:")
    print(OUTPUT_FILE)

    print()
    print("=" * 60)
    print("IMAGE QUALITY ASSESSMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()