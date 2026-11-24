from pathlib import Path
import sys

import cv2


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

from detection.lesion_detection import (
    detect_lesion
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "lesion_detection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND SAMPLE IMAGES
# ============================================================

def find_images():

    image_files = []

    for extension in [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.PNG",
        "*.JPG",
        "*.JPEG",
    ]:
        image_files.extend(
            IMAGES_PATH.rglob(extension)
        )

    return sorted(
        set(image_files)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DERMAI LESION DETECTION")
    print("=" * 60)

    print()
    print("Image root:")
    print(IMAGES_PATH)

    image_files = find_images()

    print()
    print(
        f"Images available: {len(image_files)}"
    )

    if not image_files:
        print()
        print("ERROR: No images found.")
        return

    # --------------------------------------------------------
    # Use first 20 images for initial validation
    # --------------------------------------------------------

    sample_images = image_files[:20]

    detected_count = 0

    print()
    print(
        f"Testing {len(sample_images)} images..."
    )

    for index, image_path in enumerate(
        sample_images,
        start=1
    ):

        try:

            result = detect_lesion(
                image_path
            )

            if result["detected"]:
                detected_count += 1

            visualization = result[
                "visualization"
            ]

            output_path = (
                OUTPUT_DIR /
                f"lesion_{index:02d}.jpg"
            )

            cv2.imwrite(
                str(output_path),
                visualization
            )

            box = result[
                "bounding_box"
            ]

            print(
                f"{index:02d}. "
                f"{image_path.name} "
                f"-> "
                f"{'DETECTED' if result['detected'] else 'NOT DETECTED'}"
            )

            if box:
                print(
                    f"    Box: "
                    f"x={box['x']}, "
                    f"y={box['y']}, "
                    f"w={box['width']}, "
                    f"h={box['height']}"
                )

        except Exception as exc:

            print(
                f"{index:02d}. "
                f"{image_path.name} "
                f"-> ERROR: {exc}"
            )

    print()
    print("=" * 60)
    print("LESION DETECTION SUMMARY")
    print("=" * 60)

    print(
        f"Images tested       : {len(sample_images)}"
    )

    print(
        f"Lesions detected    : {detected_count}"
    )

    print(
        f"Not detected        : "
        f"{len(sample_images) - detected_count}"
    )

    print()
    print("Visualizations saved to:")
    print(OUTPUT_DIR)

    print()
    print("=" * 60)
    print("LESION DETECTION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()