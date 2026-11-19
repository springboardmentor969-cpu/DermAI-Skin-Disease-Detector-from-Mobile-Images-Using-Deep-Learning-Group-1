"""
Data augmentation pipelines for DermAI Milestone 2.

Provides:
- Training augmentation pipeline with geometric and photometric perturbations:
  - Random horizontal and vertical flips
  - Random rotations (+/- 30 degrees)
  - Small RandomResizedCrop
  - ColorJitter (brightness, contrast, saturation, subtle hue)
  - Optional light Gaussian blur
  - ImageNet normalization ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
- Validation and Test transforms:
  - Preserves aspect ratio with square center-crop and resize to 224x224
  - Standard ImageNet normalization matching ml/preprocessing/preprocess.py
- Denormalization utility for visualization and debugging
"""

from typing import Tuple, List, Optional
import numpy as np
import torch
from torchvision import transforms
from PIL import Image as PILImage, ImageOps

# Standard ImageNet normalization constants
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]


class ExifOrientation:
    """Corrects image orientation using EXIF tags (useful for smartphone captures)."""
    def __call__(self, img: PILImage.Image) -> PILImage.Image:
        try:
            return ImageOps.exif_transpose(img)
        except Exception:
            return img


class SquareCenterCrop:
    """
    Crops the image to a square aspect ratio around the center.
    Preserves diagnostic lesion structures without geometric distortion.
    Matches ml/preprocessing/preprocess.py.
    """
    def __call__(self, img: PILImage.Image) -> PILImage.Image:
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        return img.crop((left, top, left + min_dim, top + min_dim))


def get_train_transforms(
    img_size: int = 224,
    light_blur: bool = True
) -> transforms.Compose:
    """
    Returns torchvision transforms for the training split.
    
    Includes:
    1. EXIF orientation correction
    2. Small random resized crop (80% to 100% scale, aspect ratio 0.9 to 1.1)
    3. Random horizontal and vertical flips
    4. Random rotation (+/- 30 degrees)
    5. Photometric jitter (brightness, contrast, saturation, small hue)
    6. Optional light Gaussian blur (20% probability)
    7. Conversion to Tensor
    8. ImageNet normalization
    """
    transform_list = [
        ExifOrientation(),
        transforms.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-30, 30), fill=0),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05
        )
    ]

    if light_blur:
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                p=0.2
            )
        )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return transforms.Compose(transform_list)


def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Returns deterministic transforms for validation and test splits.
    
    1. EXIF orientation correction
    2. Center-crop to square (matching ml/preprocessing/preprocess.py)
    3. Resize to target dimension (img_size x img_size) with bicubic interpolation
    4. Convert to Tensor
    5. Standard ImageNet normalization
    """
    return transforms.Compose([
        ExifOrientation(),
        SquareCenterCrop(),
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_transforms(
    split: str,
    img_size: int = 224,
    light_blur: bool = True
) -> transforms.Compose:
    """
    Factory helper to retrieve the correct transformation pipeline.
    
    Args:
        split: 'train', 'val', or 'test'
        img_size: Target image dimension (default: 224)
        light_blur: Whether to include light blur in training augmentation
    """
    split = split.lower().strip()
    if split == "train":
        return get_train_transforms(img_size=img_size, light_blur=light_blur)
    elif split in ("val", "validation", "test"):
        return get_eval_transforms(img_size=img_size)
    else:
        raise ValueError(f"Unknown split: '{split}'. Expected 'train', 'val', or 'test'.")


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized PyTorch tensor (C, H, W) back to a displayable
    RGB NumPy array (H, W, 3) in uint8 range [0, 255].
    
    Args:
        tensor: Tensor of shape (3, H, W) normalized with ImageNet statistics.
    Returns:
        np.ndarray of shape (H, W, 3) with dtype uint8.
    """
    img = tensor.detach().cpu().clone().numpy()
    if img.ndim == 3 and img.shape[0] == 3:
        # Transpose from (C, H, W) to (H, W, C)
        img = np.transpose(img, (1, 2, 0))

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)

    img = (img * std) + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

