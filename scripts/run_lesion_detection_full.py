from pathlib import Path
import csv

from detection.lesion_detection import detect_lesion


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_IMAGES_DIR = PROJECT_ROOT / "processed_images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_CSV = OUTPUT_DIR / "lesion_detection_results.csv"


# ============================================================
# FIND ALL IMAGES
# ============================================================

def get_all_images():

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    image_paths = []

    for image_path in PROCESSED_IMAGES_DIR.rglob("*"):

        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            image_paths.append(image_path)

    image_paths.sort()

    return image_paths


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "image_path",
        "detected",
        "x",
        "y",
        "width",
        "height"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DERMAI FULL DATASET LESION DETECTION")
    print("=" * 60)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Processed images:")
    print(PROCESSED_IMAGES_DIR)

    print()
    print("Output CSV:")
    print(OUTPUT_CSV)

    print()
    print("Searching for images...")

    image_paths = get_all_images()

    total_images = len(image_paths)

    print()
    print(f"Total images found : {total_images}")

    if total_images == 0:

        print()
        print("ERROR: No images found.")
        print("Check the processed_images folder.")

        return

    print()
    print("Starting lesion detection...")
    print("-" * 60)

    detected_count = 0
    not_detected_count = 0
    error_count = 0

    results = []

    for index, image_path in enumerate(image_paths, start=1):

        try:

            # ------------------------------------------------
            # Run the COMPLETE detection pipeline
            # ------------------------------------------------

            result = detect_lesion(image_path)

            # ------------------------------------------------
            # Detection successful
            # ------------------------------------------------

            if result["detected"]:

                detected_count += 1

                bounding_box = result["bounding_box"]

                results.append({
                    "image_path": str(image_path),
                    "detected": True,
                    "x": bounding_box["x"],
                    "y": bounding_box["y"],
                    "width": bounding_box["width"],
                    "height": bounding_box["height"]
                })

                print(
                    f"[{index}/{total_images}] "
                    f"DETECTED: {image_path.name} "
                    f"| x={bounding_box['x']} "
                    f"y={bounding_box['y']} "
                    f"w={bounding_box['width']} "
                    f"h={bounding_box['height']}"
                )

            # ------------------------------------------------
            # No lesion contour found
            # ------------------------------------------------

            else:

                not_detected_count += 1

                results.append({
                    "image_path": str(image_path),
                    "detected": False,
                    "x": "",
                    "y": "",
                    "width": "",
                    "height": ""
                })

                print(
                    f"[{index}/{total_images}] "
                    f"NOT DETECTED: {image_path.name}"
                )

        except Exception as error:

            error_count += 1

            results.append({
                "image_path": str(image_path),
                "detected": False,
                "x": "",
                "y": "",
                "width": "",
                "height": ""
            })

            print(
                f"[{index}/{total_images}] "
                f"ERROR: {image_path.name}"
            )

            print(
                f"    {type(error).__name__}: {error}"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    save_results(results)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    processed_count = detected_count + not_detected_count

    if processed_count > 0:

        detection_rate = (
            detected_count / processed_count
        ) * 100

    else:

        detection_rate = 0.0

    print()
    print("=" * 60)
    print("LESION DETECTION COMPLETE")
    print("=" * 60)

    print()
    print(f"Total images       : {total_images}")
    print(f"Processed          : {processed_count}")
    print(f"Detected           : {detected_count}")
    print(f"Not detected       : {not_detected_count}")
    print(f"Errors             : {error_count}")
    print(f"Detection rate     : {detection_rate:.2f}%")

    print()
    print(f"Results saved to:")
    print(OUTPUT_CSV)

    print()
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()