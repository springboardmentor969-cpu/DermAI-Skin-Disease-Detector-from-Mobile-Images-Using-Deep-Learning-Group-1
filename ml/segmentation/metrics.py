"""
Evaluation metrics for lesion segmentation:
- Dice coefficient
- IoU (Intersection over Union / Jaccard Index)
- Pixel Accuracy
- Sensitivity (Recall)
- Specificity (True Negative Rate)
"""

from typing import Dict, Union, Tuple, Optional
import torch
import numpy as np


def calculate_segmentation_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Computes segmentation metrics for binary predictions and targets.

    Args:
        pred: Predicted probabilities or raw logits (Tensor or ndarray).
        target: Binary ground-truth masks (0 or 1).
        threshold: Decision threshold for binarizing predictions.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Dict with keys: 'dice', 'iou', 'pixel_accuracy', 'sensitivity', 'specificity'
    """
    if isinstance(pred, np.ndarray):
        pred_t = torch.from_numpy(pred)
    else:
        pred_t = pred.detach().cpu()

    if isinstance(target, np.ndarray):
        target_t = torch.from_numpy(target)
    else:
        target_t = target.detach().cpu()

    # If predictions are logits (min < 0 or max > 1), apply sigmoid
    if pred_t.min() < 0.0 or pred_t.max() > 1.0:
        pred_probs = torch.sigmoid(pred_t)
    else:
        pred_probs = pred_t

    pred_bin = (pred_probs >= threshold).long().view(-1)
    target_bin = (target_t >= 0.5).long().view(-1)

    tp = ((pred_bin == 1) & (target_bin == 1)).sum().item()
    fp = ((pred_bin == 1) & (target_bin == 0)).sum().item()
    fn = ((pred_bin == 0) & (target_bin == 1)).sum().item()
    tn = ((pred_bin == 0) & (target_bin == 0)).sum().item()

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    sensitivity = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "pixel_accuracy": float(pixel_acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


class SegmentationMeter:
    """
    Accumulator for segmentation metrics across batches/epochs.
    """

    def __init__(self, threshold: float = 0.5, eps: float = 1e-6):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.total_loss = 0.0
        self.num_batches = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss: Optional[float] = None
    ) -> None:
        if pred.min() < 0.0 or pred.max() > 1.0:
            pred_probs = torch.sigmoid(pred)
        else:
            pred_probs = pred

        pred_bin = (pred_probs >= self.threshold).long().view(-1)
        target_bin = (target >= 0.5).long().view(-1)

        self.tp += ((pred_bin == 1) & (target_bin == 1)).sum().item()
        self.fp += ((pred_bin == 1) & (target_bin == 0)).sum().item()
        self.fn += ((pred_bin == 0) & (target_bin == 1)).sum().item()
        self.tn += ((pred_bin == 0) & (target_bin == 0)).sum().item()

        if loss is not None:
            self.total_loss += loss
            self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        eps = self.eps
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn

        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        sensitivity = (tp + eps) / (tp + fn + eps)
        specificity = (tn + eps) / (tn + fp + eps)
        avg_loss = (self.total_loss / max(1, self.num_batches)) if self.num_batches > 0 else 0.0

        return {
            "loss": float(avg_loss),
            "dice": float(dice),
            "iou": float(iou),
            "pixel_accuracy": float(pixel_acc),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
        }
