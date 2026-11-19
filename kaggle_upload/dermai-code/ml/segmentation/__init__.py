"""
DermAI Lesion Segmentation Module (Milestone 3).

Provides:
- ISIC 2018 Task 1 Dataset with identical image & mask augmentations.
- U-Net with pretrained ResNet34 encoder (segmentation_models_pytorch).
- Joint Dice + BCE loss.
- Segmentation metrics (Dice, IoU, Pixel Accuracy, Sensitivity, Specificity).
- Training pipeline with curves and 8-example test visualization.
- Lesion segmentation inference & ROI cropping (segment.py).
"""

from .dataset import ISICSegmentationDataset, get_segmentation_transforms, split_dataset
from .model import build_segmentation_unet
from .losses import DiceLoss, DiceBCELoss
from .metrics import calculate_segmentation_metrics, SegmentationMeter
from .segment import segment_lesion, SegmentationResult

__all__ = [
    "ISICSegmentationDataset",
    "get_segmentation_transforms",
    "split_dataset",
    "build_segmentation_unet",
    "DiceLoss",
    "DiceBCELoss",
    "calculate_segmentation_metrics",
    "SegmentationMeter",
    "segment_lesion",
    "SegmentationResult",
]
