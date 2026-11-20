# DermAI Milestone 2 (Week 4)

## Separate analysis tasks

- **Image quality assessment:** A deterministic, heuristic 0-100 technical score based on measured sharpness (Laplacian variance), exposure, contrast, and image resolution. It does not measure diagnostic quality and is not trained as a disease classifier.
- **Disease classification:** Transfer-learning classifiers for the seven HAM10000 labels using ImageNet-pretrained MobileNetV2 and EfficientNetB0. The best validation macro F1-score selects the classifier used for uploaded-image predictions. Confidence is the selected model's softmax probability.
- **Lesion segmentation:** Separate U-Net training is implemented for the publicly annotated [ISIC 2018 Task 1](https://challenge.isic-archive.com/data/#2018) dataset. HAM10000 is never used as a segmentation target, and classification output is not lesion detection. Attention U-Net is an optional comparison when resources permit.

HAM10000 contains dermoscopic images, not ordinary mobile-phone photographs. Class predictions and confidence values are research outputs, not medical diagnoses or advice.

## Leakage-safe splits

The trainer uses `StratifiedGroupKFold` with connected groups formed from `lesion_id` and exact SHA-256 duplicate-image matches. One fifth of the groups is held out for testing; one quarter of the remaining groups is used for validation. This yields approximately 60% training, 20% validation, and 20% test images. Neither a lesion ID nor an exact duplicate image crosses splits. The split manifest records every image ID and split assignment.

Both models use the same fixed seed and split. Validation macro F1-score (then validation accuracy for ties) selects the model. The held-out test set is not used for model selection and is evaluated only for the selected architecture. Accuracy, macro/weighted and per-class precision, recall and F1-score are written for validation for both architectures and for the selected model's test set. Displayed softmax confidence is not calibrated.

## Train and evaluate

From the `backend` directory, install `requirements.txt`, then run:

```powershell
python -m pip install -r requirements.txt
python train_models.py --epochs 20 --batch-size 32 --patience 4
```

For a new run, always provide separate empty report and model directories.
The trainer rejects the existing published model directory and any non-empty
run directory, so an interrupted or repeated run cannot overwrite prior
weights or reports:

```powershell
python train_models.py `
  --epochs 10 `
  --batch-size 8 `
  --patience 4 `
  --output-directory ..\reports\milestone2_run_20261102 `
  --model-directory artifacts\models\milestone2_run_20261102
```

The trainer uses ImageNet-pretrained weights, inverse-frequency class-weighted cross-entropy (normalized to mean weight 1), RandomResizedCrop, flips, rotation, mild ColorJitter, AdamW, and ReduceLROnPlateau. It records train/validation loss, accuracy, validation macro precision/recall/F1, and learning rate for every epoch. The best validation macro-F1 checkpoint is saved; validation macro-F1 and then validation accuracy are the only selection signals. Early stopping is validation-only and never reads the test set. `--epochs` is capped at 20, and `--patience` controls the number of non-improving validation epochs allowed.

The evaluation report, split manifest, and confusion-matrix CSVs are written
to the explicitly supplied report directory. New weights and the selected-model
manifest are written to the explicitly supplied model directory; existing
directories must remain untouched. ImageNet weights are downloaded on the first
run when not already cached. `--no-pretrained` disables that download, but
generally requires more training to achieve useful results.

The API's `GET /api/v1/analysis/status` endpoint reports classifier availability and the independent segmentation status. Upload responses include quality components, prediction class/confidence (or an explicit not-trained status), and segmentation status.

## Train lesion segmentation

The extracted ISBI 2016 ISIC Part 1 dataset contains 900 JPEG images and 900
PNG ground-truth masks. Keep its nested layout unchanged:

```text
segmentation_dataset\
  images\ISBI2016_ISIC_Part1_Training_Data\
  masks\ISBI2016_ISIC_Part1_Training_GroundTruth\
```

Masks use the `<image_id>_Segmentation.png` naming convention; the pipeline
normalizes that suffix when pairing. Training is independent from HAM10000
classification:

```powershell
cd backend
python train_segmentation.py --dataset-directory C:\data\segmentation_dataset
# Add --evaluate-attention to compare Attention U-Net when resources permit.
```

The command writes `backend/artifacts/segmentation/` weights and selection
metadata plus `reports`-style JSON/Markdown metrics (validation Dice selects
the model; held-out test Dice and IoU are evaluated afterward). Until this
command completes, the API explicitly reports segmentation as pending.
When available, upload responses include a generated binary mask and a
red-overlay visualization. No segmentation result is fabricated from HAM10000.
