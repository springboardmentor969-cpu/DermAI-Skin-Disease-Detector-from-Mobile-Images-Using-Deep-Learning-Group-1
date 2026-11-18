
from PIL import Image, ImageStat
from pathlib import Path
import csv

def assess_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.load()

            width, height = img.size
            image_format = img.format
            rgb = img.convert("RGB")

            stats = ImageStat.Stat(rgb)
            brightness = sum(stats.mean) / 3

            if width < 100 or height < 100:
                quality = "Low"
                message = "Resolution too small"
            elif brightness < 40:
                quality = "Low"
                message = "Image too dark"
            elif brightness > 220:
                quality = "Low"
                message = "Image too bright"
            else:
                quality = "Acceptable"
                message = "Passed basic quality checks"

            return {
                "filename": Path(image_path).name,
                "width": width,
                "height": height,
                "format": image_format,
                "brightness": round(brightness, 2),
                "quality": quality,
                "message": message
            }

    except Exception as error:
        return {
            "filename": Path(image_path).name,
            "quality": "Invalid",
            "message": str(error)
        }


def run_assessment():
    project_path = Path(__file__).resolve().parent.parent
    image_folder = project_path / "dataset" / "images"
    output_folder = project_path / "results"

    output_folder.mkdir(exist_ok=True)

    image_files = list(image_folder.rglob("*.jpg"))

    if not image_files:
        print("No JPG images found. Check dataset/images.")
        return

    output_file = output_folder / "image_quality_results.csv"

    print("DERMAI - IMAGE QUALITY ASSESSMENT")
    print("----------------------------------")
    print("Images found:", len(image_files))

    accepted = 0
    rejected = 0

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename", "width", "height", "format",
                "brightness", "quality", "message"
            ]
        )
        writer.writeheader()

        for index, image_path in enumerate(image_files, start=1):
            result = assess_image(image_path)
            writer.writerow(result)

            if result["quality"] == "Acceptable":
                accepted += 1
            else:
                rejected += 1

            if index % 500 == 0:
                print("Processed:", index)

    print("\nASSESSMENT COMPLETED")
    print("Total images:", len(image_files))
    print("Acceptable:", accepted)
    print("Low/Invalid:", rejected)
    print("Results saved to:", output_file)


if __name__ == "__main__":
    run_assessment()
