"""
Loss functions for binary lesion segmentation:
- Dice Loss
- Combined Dice + Binary Cross-Entropy (BCE) Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Operates on logits or probabilities.
    """

    def __init__(self, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Tensor of shape (B, 1, H, W) or (B, H, W)
            target: Tensor of same shape as pred, values in {0, 1}
        """
        if self.from_logits:
            pred = torch.sigmoid(pred)

        pred = pred.view(-1)
        target = target.view(-1).float()

        intersection = (pred * target).sum()
        cardinality = pred.sum() + target.sum()

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross Entropy Loss.
    Loss = dice_weight * DiceLoss + bce_weight * BCEWithLogitsLoss
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth: float = 1e-6,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth, from_logits=True)
        if pos_weight != 1.0:
            pw = torch.tensor([pos_weight], dtype=torch.float32)
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce
