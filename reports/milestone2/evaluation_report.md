# DermAI Milestone 2 model evaluation

- Dataset: HAM10000, 10015 real dermoscopic images.
- Classes: akiec, bcc, bkl, df, mel, nv, vasc.
- Device: cpu.
- Training: ImageNet-pretrained weights; final feature stage and classifier head fine-tuned for 1 epoch per model with batch size 16.
- Split: StratifiedGroupKFold: 20% held-out test fold, then 25% of the remaining 80% for validation (approximately 60/20/20); lesion_id and exact duplicate-image groups never cross splits.
- Split sizes: {'train': 6007, 'validation': 1994, 'test': 2014}.
- Model selection: highest validation macro F1-score; test metrics were not used to select the model.
- Selected model: **mobilenet_v2**.

| Model | Val accuracy | Val macro precision | Val macro recall | Val macro F1 | Test accuracy | Test macro precision | Test macro recall | Test macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mobilenet_v2 | 0.6505 | 0.3813 | 0.4664 | 0.4093 | 0.6460 | 0.3915 | 0.4489 | 0.4038 |
| efficientnet_b0 | 0.5527 | 0.3462 | 0.4094 | 0.3506 | 0.5581 | 0.3495 | 0.4197 | 0.3566 |

## Confusion matrices

Per-model validation and held-out test confusion matrices are saved as CSV files alongside this report. Full per-class precision, recall, F1-score, and support are in `evaluation_report.json`.

## Lesion detection status

Pending. HAM10000 provides diagnosis labels and lesion IDs, but no pixel-level masks or bounding boxes. Classification outputs are not lesion detections.

## Limitations

HAM10000 contains dermoscopic images, not ordinary mobile-phone photographs. The predicted class and confidence are research outputs and are not a diagnosis.
