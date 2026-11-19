"""
Threshold Calibration for Skin Lesion Quality Assessment.

1. Samples 1,000 random images from ml/data/raw/images/.
2. Computes Laplacian variance (blur), brightness, and contrast for each image.
3. Saves the metrics to ml/preprocessing/calibration_scores.csv.
4. Generates histogram plots to ml/preprocessing/calibration_histograms.png.
5. Prints percentiles (1st, 5th, 10th, 50th, 90th, 95th, 99th).
6. Determines empirical thresholds targeting:
   - 90% - 95% 'good'
   - ~3% - 7% 'acceptable'
   - Extreme outliers (< 2%) 'poor'.
7. Validates detection sensitivity on 50 artificially blurred, darkened, and overexposed images.
"""

import os
import sys
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

# Ensure root directory is on Python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Fixed random seed for exact reproducibility
RANDOM_SEED = 42
SAMPLE_SIZE = 1000

RAW_IMAGES_DIR = ROOT_DIR / "ml" / "data" / "raw" / "images"
OUTPUT_CSV = ROOT_DIR / "ml" / "preprocessing" / "calibration_scores.csv"
OUTPUT_PLOT = ROOT_DIR / "ml" / "preprocessing" / "calibration_histograms.png"


def run_calibration():
    print(f"=== DermAI Quality Threshold Calibration ===")
    print(f"Scanning raw images directory: {RAW_IMAGES_DIR}")

    image_paths = sorted(list(RAW_IMAGES_DIR.glob("*.jpg")))
    total_found = len(image_paths)
    print(f"Total HAM10000 images available: {total_found}")

    if total_found == 0:
        raise FileNotFoundError(f"No .jpg images found in {RAW_IMAGES_DIR}")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    sample_size = min(SAMPLE_SIZE, total_found)
    selected_paths = random.sample(image_paths, sample_size)
    print(f"Selected random sample of {sample_size} images (seed={RANDOM_SEED})...")

    records = []
    for idx, path in enumerate(selected_paths):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Blur score (Laplacian variance)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Brightness (mean pixel intensity, 0-255)
        brightness = float(np.mean(gray))

        # 3. Contrast (standard deviation of pixel intensity / RMS contrast)
        contrast = float(np.std(gray))

        records.append({
            "image_id": path.stem,
            "filename": path.name,
            "width": w,
            "height": h,
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
        })

        if (idx + 1) % 200 == 0 or (idx + 1) == sample_size:
            print(f"  Processed {idx + 1}/{sample_size} images...")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Saved {len(df)} score records to {OUTPUT_CSV}")

    # Calculate Percentiles
    percentiles = [1, 5, 10, 50, 90, 95, 99]
    metrics = ["blur_score", "brightness", "contrast"]

    print("\n" + "=" * 80)
    print("HAM10000 EMPIRICAL PERCENTILE DISTRIBUTION (N=1,000)")
    print("=" * 80)
    pct_summary = {}
    for m in metrics:
        vals = df[m].values
        pcts = {p: float(np.percentile(vals, p)) for p in percentiles}
        pct_summary[m] = pcts
        min_v = float(np.min(vals))
        max_v = float(np.max(vals))
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))

        print(f"\nMetric: {m.upper()}")
        print(f"  Mean: {mean_v:.2f} | Std: {std_v:.2f} | Min: {min_v:.2f} | Max: {max_v:.2f}")
        print("  Percentiles:")
        for p in percentiles:
            print(f"    {p:>2}th percentile: {pcts[p]:>8.2f}")

    # Generate and save Histograms
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(wspace=0.3)

    # Plot 1: Blur
    axes[0].hist(df["blur_score"], bins=50, color="#0284c7", edgecolor="#0369a1", alpha=0.85)
    axes[0].axvline(pct_summary["blur_score"][1], color="#dc2626", linestyle="--", label=f"1st pctl ({pct_summary['blur_score'][1]:.1f})")
    axes[0].axvline(pct_summary["blur_score"][5], color="#f59e0b", linestyle="--", label=f"5th pctl ({pct_summary['blur_score'][5]:.1f})")
    axes[0].axvline(pct_summary["blur_score"][50], color="#10b981", linestyle="-", label=f"Median ({pct_summary['blur_score'][50]:.1f})")
    axes[0].set_title("Blur Score (Laplacian Variance)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Laplacian Variance (higher = sharper)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Plot 2: Brightness
    axes[1].hist(df["brightness"], bins=50, color="#0d9488", edgecolor="#0f766e", alpha=0.85)
    axes[1].axvline(pct_summary["brightness"][1], color="#dc2626", linestyle="--", label=f"1st pctl ({pct_summary['brightness'][1]:.1f})")
    axes[1].axvline(pct_summary["brightness"][5], color="#f59e0b", linestyle="--", label=f"5th pctl ({pct_summary['brightness'][5]:.1f})")
    axes[1].axvline(pct_summary["brightness"][50], color="#10b981", linestyle="-", label=f"Median ({pct_summary['brightness'][50]:.1f})")
    axes[1].axvline(pct_summary["brightness"][95], color="#f59e0b", linestyle=":", label=f"95th pctl ({pct_summary['brightness'][95]:.1f})")
    axes[1].axvline(pct_summary["brightness"][99], color="#dc2626", linestyle=":", label=f"99th pctl ({pct_summary['brightness'][99]:.1f})")
    axes[1].set_title("Brightness (Mean Pixel Value)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Mean Grayscale Value (0-255)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    # Plot 3: Contrast
    axes[2].hist(df["contrast"], bins=50, color="#6366f1", edgecolor="#4f46e5", alpha=0.85)
    axes[2].axvline(pct_summary["contrast"][1], color="#dc2626", linestyle="--", label=f"1st pctl ({pct_summary['contrast'][1]:.1f})")
    axes[2].axvline(pct_summary["contrast"][5], color="#f59e0b", linestyle="--", label=f"5th pctl ({pct_summary['contrast'][5]:.1f})")
    axes[2].axvline(pct_summary["contrast"][50], color="#10b981", linestyle="-", label=f"Median ({pct_summary['contrast'][50]:.1f})")
    axes[2].set_title("Contrast (RMS Intensity Std)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Standard Deviation (0-127)")
    axes[2].set_ylabel("Frequency")
    axes[2].legend(fontsize=9)
    axes[2].grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("HAM10000 Skin Lesion Image Quality Distribution (N=1,000 Sample)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Histograms saved to: {OUTPUT_PLOT}")

    # Determine Calibrated Thresholds
    # Goal: ~90-95% "good", ~3-7% "acceptable", extreme outliers (< 2%) "poor".
    #
    # Blur:
    #   1st percentile is extreme outlier (poor): e.g. < 15.0
    #   5th percentile is borderline (acceptable): e.g. 15.0 - 22.0
    #   > 22.0 is good (> 95% of images)
    #
    # Brightness:
    #   1st percentile dark (poor): < 100.0 (HAM10000 min is ~105-110)
    #   5th percentile dark (acceptable): 100.0 - 120.0
    #   95th percentile bright (acceptable): 185.0 - 210.0
    #   99th percentile bright (poor): > 210.0
    #
    # Contrast:
    #   1st percentile (poor): < 8.0
    #   5th percentile (acceptable): 8.0 - 12.0
    #   > 12.0 is good (> 95% of images)

    blur_p1 = pct_summary["blur_score"][1]
    blur_p5 = pct_summary["blur_score"][5]

    bright_p1 = pct_summary["brightness"][1]
    bright_p5 = pct_summary["brightness"][5]
    bright_p95 = pct_summary["brightness"][95]
    bright_p99 = pct_summary["brightness"][99]

    contrast_p1 = pct_summary["contrast"][1]
    contrast_p5 = pct_summary["contrast"][5]

    # Calibrated thresholds derived from empirical percentiles:
    # - Blur: 1st percentile is 15.35, 5th is 21.79. Setting poor=15.0, acceptable=20.0
    # - Brightness: 1st pctl is 118.23, 99th is 199.67. Setting dark_poor=100.0, dark_acc=115.0, bright_acc=195.0, bright_poor=210.0
    # - Contrast: 1st pctl is 9.96, 5th is 13.30. Setting poor=8.0, acceptable=11.0
    # - Min dimension: poor=224, acceptable=300 (standard dermoscopy is 450x600)
    calibrated_thresholds = {
        "blur_poor": 15.0,
        "blur_acceptable": 20.0,
        "brightness_dark_poor": 100.0,
        "brightness_dark_acceptable": 115.0,
        "brightness_bright_acceptable": 195.0,
        "brightness_bright_poor": 210.0,
        "contrast_poor": 8.0,
        "contrast_acceptable": 11.0,
        "min_dimension_poor": 224,
        "min_dimension_acceptable": 300,
    }

    # Simulate classification on the 1000 sample images
    def classify(row, t):
        b = row["blur_score"]
        br = row["brightness"]
        c = row["contrast"]
        w = row["width"]
        h = row["height"]

        # Check poor
        if (b < t["blur_poor"] or
            br < t["brightness_dark_poor"] or
            br > t["brightness_bright_poor"] or
            c < t["contrast_poor"] or
            min(w, h) < t["min_dimension_poor"]):
            return "poor"

        # Check acceptable
        if (b < t["blur_acceptable"] or
            br < t["brightness_dark_acceptable"] or
            br > t["brightness_bright_acceptable"] or
            c < t["contrast_acceptable"] or
            min(w, h) < t["min_dimension_acceptable"]):
            return "acceptable"

        return "good"

    # Evaluate with current calibrated thresholds
    labels = df.apply(lambda row: classify(row, calibrated_thresholds), axis=1)
    counts = labels.value_counts()
    n_total = len(df)

    good_pct = (counts.get("good", 0) / n_total) * 100
    acc_pct = (counts.get("acceptable", 0) / n_total) * 100
    poor_pct = (counts.get("poor", 0) / n_total) * 100

    print("\n" + "=" * 80)
    print("PROPOSED CALIBRATED THRESHOLDS & SAMPLE EVALUATION")
    print("=" * 80)
    for k, v in calibrated_thresholds.items():
        print(f"  '{k}': {v}")

    print(f"\nSample Classification Distribution (Target: 90-95% Good):")
    print(f"  GOOD:       {counts.get('good', 0):>4} ({good_pct:>5.1f}%)  [Target: 90% - 95%]")
    print(f"  ACCEPTABLE: {counts.get('acceptable', 0):>4} ({acc_pct:>5.1f}%)  [Target: ~3% - 7%]")
    print(f"  POOR:       {counts.get('poor', 0):>4} ({poor_pct:>5.1f}%)  [Target: < 2%]")

    # Run Degradation Sensitivity Test (50 artificially blurred, darkened, brightened)
    run_degradation_test(selected_paths[:50], calibrated_thresholds)

    return calibrated_thresholds, pct_summary


def run_degradation_test(test_image_paths, thresholds):
    """
    Creates test images by artificially blurring (Gaussian blur),
    darkening, and brightening 50 good images, and verifies detection accuracy.
    """
    print("\n" + "=" * 80)
    print("DEGRADATION SENSITIVITY VALIDATION (50 IMAGES PER CONDITION)")
    print("=" * 80)

    from ml.preprocessing.quality import assess_image_quality

    results = {
        "original_good": {"good": 0, "acceptable": 0, "poor": 0},
        "blurred": {"good": 0, "acceptable": 0, "poor": 0},
        "darkened": {"good": 0, "acceptable": 0, "poor": 0},
        "overexposed": {"good": 0, "acceptable": 0, "poor": 0},
    }

    for path in test_image_paths:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            continue

        # 1. Original
        res_orig = assess_image_quality(img_bgr, thresholds=thresholds)
        results["original_good"][res_orig["quality_label"]] += 1

        # 2. Artificially Blurred (Gaussian Blur: kernel 21x21, sigma 6)
        blurred_bgr = cv2.GaussianBlur(img_bgr, (21, 21), 6.0)
        res_blur = assess_image_quality(blurred_bgr, thresholds=thresholds)
        results["blurred"][res_blur["quality_label"]] += 1

        # 3. Artificially Darkened (scale brightness to 20%)
        darkened_bgr = np.clip(img_bgr * 0.20, 0, 255).astype(np.uint8)
        res_dark = assess_image_quality(darkened_bgr, thresholds=thresholds)
        results["darkened"][res_dark["quality_label"]] += 1

        # 4. Artificially Overexposed / Washed out (increase intensity by +100)
        overexposed_bgr = np.clip(img_bgr.astype(np.float32) + 105.0, 0, 255).astype(np.uint8)
        res_over = assess_image_quality(overexposed_bgr, thresholds=thresholds)
        results["overexposed"][res_over["quality_label"]] += 1

    print("\n+------------------------------+------------+-----------------+------------+------------------+")
    print("| Test Condition               | Total Imgs | Good / Acceptable | Poor (Flagged) | Detection Rate   |")
    print("+------------------------------+------------+-----------------+------------+------------------+")

    # Original
    orig_ok = results["original_good"]["good"] + results["original_good"]["acceptable"]
    orig_poor = results["original_good"]["poor"]
    print(f"| Original Clean Images        | {len(test_image_paths):>10} | {orig_ok:>15} | {orig_poor:>10} | {orig_ok/len(test_image_paths)*100:>15.1f}% |")

    # Blurred
    blur_ok = results["blurred"]["good"] + results["blurred"]["acceptable"]
    blur_poor = results["blurred"]["poor"]
    print(f"| Artificially Blurred         | {len(test_image_paths):>10} | {blur_ok:>15} | {blur_poor:>10} | {blur_poor/len(test_image_paths)*100:>15.1f}% |")

    # Darkened
    dark_ok = results["darkened"]["good"] + results["darkened"]["acceptable"]
    dark_poor = results["darkened"]["poor"]
    print(f"| Artificially Darkened        | {len(test_image_paths):>10} | {dark_ok:>15} | {dark_poor:>10} | {dark_poor/len(test_image_paths)*100:>15.1f}% |")

    # Overexposed
    over_ok = results["overexposed"]["good"] + results["overexposed"]["acceptable"]
    over_poor = results["overexposed"]["poor"]
    print(f"| Artificially Overexposed     | {len(test_image_paths):>10} | {over_ok:>15} | {over_poor:>10} | {over_poor/len(test_image_paths)*100:>15.1f}% |")
    print("+------------------------------+------------+-----------------+------------+------------------+")

    # Combined defect detection rate
    total_degraded = len(test_image_paths) * 3
    detected_degraded = blur_poor + dark_poor + over_poor
    print(f"\nOverall Defect Detection Sensitivity: {detected_degraded}/{total_degraded} ({detected_degraded/total_degraded*100:.1f}%)")


if __name__ == "__main__":
    run_calibration()
