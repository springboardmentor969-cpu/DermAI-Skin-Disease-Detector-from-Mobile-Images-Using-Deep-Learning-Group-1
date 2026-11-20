# DermAI dataset inspection report

## Scope and source

- The project uses the existing Kaggle HAM10000 dataset already present in this workspace under `DermAI`.
- Metadata file: `HAM10000_metadata.csv`
- Image directories: HAM10000_images_part_1, HAM10000_images_part_2

## Data overview

- Total image files discovered: 10015
- Supported image formats: {'.jpg': 10015}
- Duplicate file groups found: 2
- Corrupted/unreadable files: 0

## Class labels and counts

- `akiec`: 327
- `bcc`: 514
- `bkl`: 1099
- `df`: 115
- `mel`: 1113
- `nv`: 6705
- `vasc`: 142

## Metadata quality checks

- Missing values: {'lesion_id': 0, 'image_id': 0, 'dx': 0, 'dx_type': 0, 'age': 57, 'sex': 0, 'localization': 0}
- Age statistics: {'count': 9958.0, 'mean': 51.863828077927295, 'std': 16.968613692495364, 'min': 0.0, '25%': 40.0, '50%': 50.0, '75%': 65.0, 'max': 85.0}
- Sex counts: {'male': 5406, 'female': 4552, 'unknown': 57}
- Diagnosis types: {'histo': 5340, 'follow_up': 3704, 'consensus': 902, 'confocal': 69}

## Notes

- The dataset is dermoscopic, not a typical mobile-clinical photograph set; imaging metadata indicates skin lesion images collected from dermoscopy workflows.
- The dataset is class-imbalanced, with `nv` as the dominant label and smaller classes such as `df` and `vasc` underrepresented.
- Patient and lesion identifiers are preserved in metadata for future leakage-safe splitting and pre-processing design.
- No training or validation splits were created during inspection; the output is limited to dataset characterization and reporting.

## File outputs

- `reports/class_distribution.csv`
- `reports/class_distribution.png`
- `reports/dataset_report.md`