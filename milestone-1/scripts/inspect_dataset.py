from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_PATH = ROOT_DIR / "HAM10000_metadata.csv"
IMAGE_DIRS = [ROOT_DIR / "HAM10000_images_part_1", ROOT_DIR / "HAM10000_images_part_2"]


def collect_image_paths() -> list[Path]:
    image_paths: list[Path] = []
    for image_dir in IMAGE_DIRS:
        if image_dir.exists():
            image_paths.extend(
                [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}]
            )
    return sorted(image_paths)


def read_metadata() -> pd.DataFrame:
    return pd.read_csv(METADATA_PATH)


def detect_duplicates(image_paths: list[Path]) -> tuple[int, list[str]]:
    seen: dict[str, list[str]] = {}
    for path in image_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        seen.setdefault(digest, []).append(str(path.relative_to(ROOT_DIR)))
    dup_groups = {digest: files for digest, files in seen.items() if len(files) > 1}
    duplicate_count = sum(len(files) for files in dup_groups.values())
    return len(dup_groups), [item for files in dup_groups.values() for item in files]


def inspect_images(image_paths: list[Path]) -> dict[str, object]:
    formats = Counter()
    dimensions = Counter()
    corrupt_files: list[str] = []
    for path in image_paths:
        formats[path.suffix.lower()] += 1
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions[(image.size[0], image.size[1], image.mode)] += 1
        except Exception:
            corrupt_files.append(str(path.relative_to(ROOT_DIR)))
    return {
        "total_images": len(image_paths),
        "formats": dict(formats),
        "dimensions": dict(dimensions),
        "corrupt_files": corrupt_files,
    }


def write_report(metadata: pd.DataFrame, image_summary: dict[str, object], duplicate_count: int) -> None:
    class_counts = metadata["dx"].value_counts().sort_index()
    csv_path = OUTPUT_DIR / "class_distribution.csv"
    class_counts.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    class_counts.plot(kind="bar", ax=ax, color=["#2E86DE", "#7CB342", "#F39C12", "#E74C3C", "#8E44AD", "#16A085", "#D35400"])
    ax.set_title("HAM10000 class distribution")
    ax.set_xlabel("Diagnosis class")
    ax.set_ylabel("Image count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)

    metadata_missing = metadata.isna().sum().to_dict()
    report_lines = [
        "# DermAI dataset inspection report",
        "",
        "## Scope and source",
        "",
        f"- The project uses the existing Kaggle HAM10000 dataset already present in this workspace under `{ROOT_DIR.name}`.",
        f"- Metadata file: `{METADATA_PATH.name}`",
        f"- Image directories: {', '.join(p.name for p in IMAGE_DIRS if p.exists())}",
        "",
        "## Data overview",
        "",
        f"- Total image files discovered: {image_summary['total_images']}",
        f"- Supported image formats: {image_summary['formats']}",
        f"- Duplicate file groups found: {duplicate_count}",
        f"- Corrupted/unreadable files: {len(image_summary['corrupt_files'])}",
        "",
        "## Class labels and counts",
        "",
    ]
    for label, count in class_counts.items():
        report_lines.append(f"- `{label}`: {count}")

    report_lines.extend([
        "",
        "## Metadata quality checks",
        "",
        f"- Missing values: {metadata_missing}",
        f"- Age statistics: {metadata['age'].describe().to_dict()}",
        f"- Sex counts: {metadata['sex'].value_counts(dropna=False).to_dict()}",
        f"- Diagnosis types: {metadata['dx_type'].value_counts().to_dict()}",
        "",
        "## Notes",
        "",
        "- The dataset is dermoscopic, not a typical mobile-clinical photograph set; imaging metadata indicates skin lesion images collected from dermoscopy workflows.",
        "- The dataset is class-imbalanced, with `nv` as the dominant label and smaller classes such as `df` and `vasc` underrepresented.",
        "- Patient and lesion identifiers are preserved in metadata for future leakage-safe splitting and pre-processing design.",
        "- No training or validation splits were created during inspection; the output is limited to dataset characterization and reporting.",
        "",
        "## File outputs",
        "",
        "- `reports/class_distribution.csv`",
        "- `reports/class_distribution.png`",
        "- `reports/dataset_report.md`",
    ])

    (OUTPUT_DIR / "dataset_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    image_paths = collect_image_paths()
    metadata = read_metadata()
    image_summary = inspect_images(image_paths)
    duplicate_group_count, _ = detect_duplicates(image_paths)
    write_report(metadata, image_summary, duplicate_group_count)
    print(f"Discovered {len(image_paths)} image files.")
    print(f"Metadata rows: {len(metadata)}")
    print(f"Class counts: {metadata['dx'].value_counts().to_dict()}")
    print(f"Known corrupt files: {len(image_summary['corrupt_files'])}")
    print(f"Report at {OUTPUT_DIR / 'dataset_report.md'}")


if __name__ == "__main__":
    main()
