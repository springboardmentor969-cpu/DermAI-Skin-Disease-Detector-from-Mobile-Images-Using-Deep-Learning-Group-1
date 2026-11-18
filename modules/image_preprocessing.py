
from PIL import Image, ImageOps
from pathlib import Path
import csv

def preprocess_image(image_path, output_path):
    try:
        with Image.open(image_path) as img:
            img.load()

            # Convert to RGB
            img = img.convert("RGB")

            # Resize while preserving a consistent input size
            img = ImageOps.fit(
                img,
                (224, 224),
                method=Image.Resampling.LANCZOS
            )

            # Basic contrast enhancement
            img = ImageOps.autocontrast(img)

            # Save processed image
            img.save(output_path, "JPEG", quality=95)

        return "Success"

    except Exception as error:
        return f"Failed: {error}"


def run_preprocessing():
    project_path = Path(__file__).resolve().parent.parent

    input_folder = project_path / "dataset" / "images"
    output_folder = project_path / "dataset" / "preprocessed_images"
    results_folder = project_path / "results"

    output_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png"}
    image_files = [
        p for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    print("DERMAI - IMAGE PREPROCESSING")
    print("--------------------------------")
    print("Images found:", len(image_files))

    successful = 0
    failed = 0

    report_path = results_folder / "preprocessing_report.csv"

    with open(report_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "status"])

        for index, image_path in enumerate(image_files, start=1):
            output_path = output_folder / f"{image_path.stem}.jpg"

            status = preprocess_image(image_path, output_path)
            writer.writerow([image_path.name, status])

            if status == "Success":
                successful += 1
            else:
                failed += 1

            if index % 500 == 0:
                print("Processed:", index)

    print("\nPREPROCESSING COMPLETED")
    print("Total:", len(image_files))
    print("Successful:", successful)
    print("Failed:", failed)
    print("Saved images:", output_folder)
    print("Report:", report_path)


if __name__ == "__main__":
    run_preprocessing()
