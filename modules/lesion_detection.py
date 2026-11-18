
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import csv

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "dataset" / "preprocessed_images"
OUTPUT_DIR = BASE_DIR / "dataset" / "lesion_detection_results"
REPORT_DIR = BASE_DIR / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

report = []
image_files = []

for extension in ("*.jpg", "*.jpeg", "*.png"):
    image_files.extend(INPUT_DIR.rglob(extension))

print("DERMAI - LESION REGION DETECTION")
print("Images found:", len(image_files))

for index, image_path in enumerate(image_files, start=1):
    try:
        image = Image.open(image_path).convert("RGB")
        gray = image.convert("L").filter(ImageFilter.MedianFilter(3))

        # Select darker pixels as a basic experimental region proposal
        histogram = gray.histogram()
        total = sum(histogram)
        target = int(total * 0.25)
        cumulative = 0
        threshold = 0

        for value, count in enumerate(histogram):
            cumulative += count
            if cumulative >= target:
                threshold = value
                break

        mask = gray.point(
            lambda pixel: 255 if pixel <= threshold else 0
        )

        bbox = mask.getbbox()

        if bbox:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            detected = 1 if area > 0 else 0

            overlay = Image.new("RGB", image.size, (255, 0, 0))
            highlighted = Image.composite(overlay, image, mask)
            result = Image.blend(image, highlighted, 0.35)

            draw = ImageDraw.Draw(result)
            draw.rectangle(bbox, outline="yellow", width=2)
        else:
            detected = 0
            result = image.copy()
            x1 = y1 = x2 = y2 = 0

        output_path = OUTPUT_DIR / f"{image_path.stem}_detected.jpg"
        result.save(output_path, "JPEG", quality=90)

        report.append([
            image_path.name, detected,
            x1, y1, x2, y2, threshold
        ])

        if index % 500 == 0:
            print("Processed:", index)

    except Exception as error:
        report.append([image_path.name, "Error", "", "", "", "", str(error)])

report_path = REPORT_DIR / "lesion_detection_report.csv"

with open(report_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Image", "Region_Proposal",
        "X1", "Y1", "X2", "Y2", "Threshold"
    ])
    writer.writerows(report)

print("\nLESION REGION PROTOTYPE COMPLETED")
print("Total images:", len(image_files))
print("Results saved in:", OUTPUT_DIR)
print("Report saved in:", report_path)
