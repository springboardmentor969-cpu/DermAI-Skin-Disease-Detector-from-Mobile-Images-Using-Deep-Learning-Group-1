"""
Test script for DermAI image preprocessing and quality assessment pipeline.
Evaluates sample images against quality.py and preprocess.py, printing a formatted results table.
"""

import sys
from pathlib import Path
import argparse
import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.quality import assess_image_quality
from ml.preprocessing.preprocess import preprocess_image


def generate_sample_images_if_needed(directory: Path):
    """
    Generate diverse sample images representing different clinical photograph scenarios
    if the target folder is empty.
    """
    directory.mkdir(parents=True, exist_ok=True)
    existing_images = list(directory.glob("*.jpg")) + list(directory.glob("*.png"))
    if existing_images:
        return

    print(f"Generating synthetic clinical test images in: {directory} ...")

    # 1. Good quality sharp lesion
    img_good = PILImage.new("RGB", (600, 600), color=(220, 190, 175))
    draw = ImageDraw.Draw(img_good)
    # Draw simulated mole/lesion
    draw.ellipse([200, 200, 400, 400], fill=(70, 35, 20), outline=(40, 20, 10), width=3)
    draw.ellipse([260, 250, 340, 330], fill=(30, 15, 10))
    img_good.save(directory / "01_sharp_good.jpg", format="JPEG", quality=95)

    # 2. Blurry lesion (heavy Gaussian blur)
    img_blur_np = cv2.GaussianBlur(np.array(img_good), (45, 45), 0)
    PILImage.fromarray(img_blur_np).save(directory / "02_blurry_poor.jpg", format="JPEG")

    # 3. Under-exposed / Dark photo
    img_dark = PILImage.new("RGB", (500, 500), color=(25, 20, 20))
    draw_dark = ImageDraw.Draw(img_dark)
    draw_dark.ellipse([150, 150, 350, 350], fill=(12, 8, 5))
    img_dark.save(directory / "03_dark_underexposed.jpg", format="JPEG")

    # 4. Over-exposed / Washed out photo
    img_bright = PILImage.new("RGB", (500, 500), color=(245, 240, 235))
    draw_bright = ImageDraw.Draw(img_bright)
    draw_bright.ellipse([180, 180, 320, 320], fill=(225, 200, 190))
    img_bright.save(directory / "04_washed_out_overexposed.jpg", format="JPEG")

    # 5. Low contrast image
    img_low_contrast = PILImage.new("RGB", (480, 480), color=(140, 140, 140))
    draw_lc = ImageDraw.Draw(img_low_contrast)
    draw_lc.ellipse([180, 180, 300, 300], fill=(148, 148, 148))
    img_low_contrast.save(directory / "05_low_contrast.jpg", format="JPEG")

    # 6. Acceptable quality lesion
    img_acceptable = PILImage.new("RGB", (480, 480), color=(210, 180, 160))
    draw_acc = ImageDraw.Draw(img_acceptable)
    draw_acc.ellipse([180, 180, 300, 300], fill=(85, 45, 30))
    img_acceptable.save(directory / "06_acceptable_lesion.png", format="PNG")


def run_evaluation(folder_path: Path):
    generate_sample_images_if_needed(folder_path)

    image_files = sorted(
        [p for p in folder_path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    if not image_files:
        print(f"No JPG/PNG images found in {folder_path}.")
        return

    # Print Table Header
    col_file = 30
    col_res = 14
    col_blur = 12
    col_bright = 12
    col_cont = 10
    col_label = 15
    col_tensor = 20

    separator = (
        "+" + "-" * (col_file + 2)
        + "+" + "-" * (col_res + 2)
        + "+" + "-" * (col_blur + 2)
        + "+" + "-" * (col_bright + 2)
        + "+" + "-" * (col_cont + 2)
        + "+" + "-" * (col_label + 2)
        + "+" + "-" * (col_tensor + 2)
        + "+"
    )

    print("\n" + "=" * 115)
    print("DERMAI IMAGE QUALITY & PREPROCESSING EVALUATION RESULTS")
    print("=" * 115)
    print(separator)
    header = (
        f"| {'Image Filename':<{col_file}} "
        f"| {'Resolution':<{col_res}} "
        f"| {'Blur Score':<{col_blur}} "
        f"| {'Brightness':<{col_bright}} "
        f"| {'Contrast':<{col_cont}} "
        f"| {'Quality':<{col_label}} "
        f"| {'PyTorch Array':<{col_tensor}} |"
    )
    print(header)
    print(separator)

    stats = {"good": 0, "acceptable": 0, "poor": 0}

    for img_path in image_files:
        try:
            # 1. Run Quality Assessment
            q_res = assess_image_quality(img_path)
            # 2. Run Preprocessing Pipeline
            tensor_arr = preprocess_image(img_path)

            filename = img_path.name[:col_file]
            res_str = f"{q_res['resolution']['width']}x{q_res['resolution']['height']}"
            blur_str = f"{q_res['blur_score']:.1f}"
            bright_str = f"{q_res['brightness']:.1f}"
            cont_str = f"{q_res['contrast']:.1f}"
            label = q_res["quality_label"].upper()
            tensor_str = f"{tensor_arr.shape} ({tensor_arr.dtype})"

            stats[q_res["quality_label"]] += 1

            row = (
                f"| {filename:<{col_file}} "
                f"| {res_str:<{col_res}} "
                f"| {blur_str:<{col_blur}} "
                f"| {bright_str:<{col_bright}} "
                f"| {cont_str:<{col_cont}} "
                f"| {label:<{col_label}} "
                f"| {tensor_str:<{col_tensor}} |"
            )
            print(row)
        except Exception as e:
            print(f"| {img_path.name:<{col_file}} | ERROR: {str(e):<85} |")

    print(separator)
    print(f"\nSummary: Total={len(image_files)} | Good={stats['good']} | Acceptable={stats['acceptable']} | Poor={stats['poor']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image quality & preprocessing pipeline.")
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(PROJECT_ROOT / "ml" / "preprocessing" / "sample_images"),
        help="Folder containing test images",
    )
    args = parser.parse_args()
    run_evaluation(Path(args.folder))
