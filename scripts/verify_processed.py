from pathlib import Path
from PIL import Image


# ============================================================
# PROCESSED IMAGE VERIFICATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "processed_images"

EXPECTED_SIZE = (224, 224)


print("=" * 60)
print("PROCESSED IMAGE VERIFICATION")
print("=" * 60)

print("\nProcessed images:")
print(PROCESSED_DIR)


image_paths = list(PROCESSED_DIR.glob("*.png"))

print(f"\nTotal processed images found: {len(image_paths)}")


wrong_size = 0
wrong_mode = 0
corrupted = 0

dimension_counts = {}
mode_counts = {}


for image_path in image_paths:

    try:

        with Image.open(image_path) as image:

            size = image.size
            mode = image.mode

            dimension_counts[size] = (
                dimension_counts.get(size, 0) + 1
            )

            mode_counts[mode] = (
                mode_counts.get(mode, 0) + 1
            )

            if size != EXPECTED_SIZE:
                wrong_size += 1

            if mode != "RGB":
                wrong_mode += 1

    except Exception as error:

        corrupted += 1

        print(
            f"Unreadable: {image_path.name} -> {error}"
        )


print("\n" + "=" * 60)
print("DIMENSION DISTRIBUTION")
print("=" * 60)

for dimension, count in dimension_counts.items():

    print(
        f"{dimension[0]} x {dimension[1]} : {count}"
    )


print("\n" + "=" * 60)
print("IMAGE MODE DISTRIBUTION")
print("=" * 60)

for mode, count in mode_counts.items():

    print(f"{mode} : {count}")


print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)

print(f"Total images       : {len(image_paths)}")
print(f"Expected size      : 224 x 224")
print(f"Wrong dimensions   : {wrong_size}")
print(f"Wrong color mode   : {wrong_mode}")
print(f"Corrupted images   : {corrupted}")


if (
    len(image_paths) == 2298
    and wrong_size == 0
    and wrong_mode == 0
    and corrupted == 0
):

    print("\nSTATUS: PASSED")

else:

    print("\nSTATUS: CHECK REQUIRED")


print("=" * 60)