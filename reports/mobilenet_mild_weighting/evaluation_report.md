# DermAI Milestone 2 model evaluation

- Dataset: HAM10000, 10015 real dermoscopic images.
- Classes: akiec, bcc, bkl, df, mel, nv, vasc.
- Device: cpu.
- Training: ImageNet-pretrained weights; final feature stage and classifier head fine-tuned for up to 10 epoch(s) per model with batch size 8.
- Early stopping: patience=4, min_delta=0.0001; ReduceLROnPlateau halves the learning rate after one validation macro-F1 plateau.
- Split: StratifiedGroupKFold: 20% held-out test fold, then 25% of the remaining 80% for validation (approximately 60/20/20); lesion_id and exact duplicate-image groups never cross splits.
- Split sizes: {'train': 6007, 'validation': 1994, 'test': 2014}.
- Model selection: highest validation macro F1-score; test metrics were not used to select the model.
- Selected model: **mobilenet_v2**.

| Model | Best epoch | Stopped epoch | Val accuracy | Val macro precision | Val macro recall | Val macro F1 | Test accuracy | Test macro precision | Test macro recall | Test macro F1 | Overfit gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mobilenet_v2 | 10 | 10 | 0.7723 | 0.5593 | 0.6544 | 0.5950 | — | — | — | — | 0.0034 |

## Confusion matrices

Validation confusion matrices for both models and a held-out test confusion matrix for the selected model are saved as CSV files alongside this report. Full per-class precision, recall, F1-score, and support are in `evaluation_report.json`.

## Lesion detection status

Pending. HAM10000 provides diagnosis labels and lesion IDs, but no pixel-level masks or bounding boxes. Classification outputs are not lesion detections.

## Limitations

HAM10000 contains dermoscopic images, not ordinary mobile-phone photographs. The predicted class and confidence are research outputs and are not a diagnosis.
