"""
Dataset and data loading utilities for ISIC 2018 Task 1 Lesion Segmentation.

Features:
- Discovers and pairs ISIC 2018 Task 1 images (ISIC_<id>.jpg) and ground-truth masks (ISIC_<id>_segmentation.png).
- Flexible path resolution (supports flat directories, subdirectories, or ISIC challenge archive structure).
- Deterministic 80 / 10 / 10 split (train, val, test) with a fixed seed (default 42).
- Persists segmentation split to CSV for reproducibility.
- Resize to 256x256 with identical geometric augmentations (flips, rotations) on image and mask,
  plus photometric augmentation (brightness/contrast) on the image.
- Normalizes images with ImageNet statistics and outputs binary float masks (0.0 or 1.0).
- Provides a smoke sample generator from local images if the dataset is not yet downloaded.
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_segmentation_transforms(
    split: str = "train",
    img_size: int = 256
) -> A.Compose:
    """
    Returns albumentations transforms for segmentation.
    Augmentations are applied identically to image and mask for spatial transforms
    (flips, rotations, affine shifts), and to image only for pixel-level transforms (brightness/contrast).
    """
    if split.lower() == "train":
        return A.Compose([
            A.Resize(
                height=img_size,
                width=img_size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.10,
                rotate_limit=45,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(
                height=img_size,
                width=img_size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])


def discover_image_mask_pairs(
    data_dir: Union[str, Path]
) -> List[Dict[str, str]]:
    """
    Scans data_dir (recursively) to find pairs of ISIC images and segmentation masks.
    Image naming pattern: ISIC_<id>.jpg (or .png/.jpeg)
    Mask naming pattern : ISIC_<id>_segmentation.png (or .jpg)
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []

    # Map image_id -> mask_path
    masks_by_id: Dict[str, Path] = {}
    images_by_id: Dict[str, Path] = {}

    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        stem = path.stem.lower()
        # Check if it's a mask
        if "_segmentation" in stem or "_mask" in stem:
            # Extract ISIC_XXXXXXX
            match = re.search(r"(isic_\d+)", stem)
            if match:
                img_id = match.group(1).upper()
                masks_by_id[img_id] = path
        elif path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            match = re.search(r"(isic_\d+)", stem)
            if match:
                img_id = match.group(1).upper()
                images_by_id[img_id] = path

    # Intersect
    common_ids = sorted(list(set(images_by_id.keys()) & set(masks_by_id.keys())))
    pairs = [
        {
            "image_id": img_id,
            "image_path": str(images_by_id[img_id].resolve()),
            "mask_path": str(masks_by_id[img_id].resolve()),
        }
        for img_id in common_ids
    ]
    return pairs


def split_dataset(
    pairs: List[Dict[str, str]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    save_csv_path: Optional[Union[str, Path]] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Deterministically splits paired samples into train (80%), val (10%), test (10%).
    Persists to CSV if save_csv_path is specified.
    """
    train_ratio, val_ratio, test_ratio = ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # Sort deterministically by image_id
    sorted_pairs = sorted(pairs, key=lambda x: x["image_id"])
    n = len(sorted_pairs)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    # Ensure all items accounted for
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    split_dict: Dict[str, List[Dict[str, str]]] = {
        "train": [sorted_pairs[i] for i in train_indices],
        "val": [sorted_pairs[i] for i in val_indices],
        "test": [sorted_pairs[i] for i in test_indices],
    }

    if save_csv_path is not None:
        save_csv_path = Path(save_csv_path)
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for s_name, p_list in split_dict.items():
            for p in p_list:
                rows.append({
                    "image_id": p["image_id"],
                    "split": s_name,
                    "image_path": p["image_path"],
                    "mask_path": p["mask_path"],
                })
        df = pd.DataFrame(rows)
        df.to_csv(save_csv_path, index=False)

    return split_dict


def create_smoke_dataset(
    output_dir: Union[str, Path] = "ml/data/isic2018_task1",
    num_samples: int = 24,
    source_images_dir: Union[str, Path] = "ml/data/raw/images",
    seed: int = 42,
) -> Path:
    """
    Creates synthetic lesion segmentation ground-truth pairs for smoke testing
    using existing HAM10000 images if ISIC 2018 Task 1 is not yet downloaded.
    Ground truth masks are generated using color thresholding and Otsu segmentation
    to simulate real skin lesion masks.
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    src_dir = Path(source_images_dir)
    available_images = list(src_dir.glob("*.jpg"))
    if not available_images:
        # Fallback: create synthetic noise skin images
        rng = np.random.RandomState(seed)
        for i in range(num_samples):
            img_id = f"ISIC_{i:07d}"
            img = np.full((300, 400, 3), (220, 180, 170), dtype=np.uint8)
            # Add darker ellipse lesion
            center = (200 + rng.randint(-30, 30), 150 + rng.randint(-30, 30))
            axes = (rng.randint(40, 80), rng.randint(30, 60))
            angle = rng.randint(0, 180)
            cv2.ellipse(img, center, axes, angle, 0, 360, (80, 50, 40), -1)
            cv2.imwrite(str(images_dir / f"{img_id}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            mask = np.zeros((300, 400), dtype=np.uint8)
            cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
            cv2.imwrite(str(masks_dir / f"{img_id}_segmentation.png"), mask)
        return output_path

    # Sample available images
    rng = np.random.RandomState(seed)
    sampled = [available_images[i] for i in rng.choice(len(available_images), min(num_samples, len(available_images)), replace=False)]

    for img_path in sampled:
        img_id = img_path.stem
        dst_img_path = images_dir / f"{img_id}.jpg"
        dst_mask_path = masks_dir / f"{img_id}_segmentation.png"

        # Read image
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        cv2.imwrite(str(dst_img_path), bgr)

        # Generate realistic lesion mask via color difference & Otsu in HSV/Lab space
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        # Skin lesions are darker than surrounding skin
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Clean up mask (keep largest central contour)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
            else:
                # Fallback central ellipse
                h, w = gray.shape
                cv2.ellipse(mask, (w // 2, h // 2), (w // 4, h // 4), 0, 0, 360, 255, -1)
        else:
            h, w = gray.shape
            cv2.ellipse(mask, (w // 2, h // 2), (w // 4, h // 4), 0, 0, 360, 255, -1)

        cv2.imwrite(str(dst_mask_path), mask)

    return output_path


class ISICSegmentationDataset(Dataset):
    """
    PyTorch Dataset for ISIC 2018 Task 1 Skin Lesion Segmentation.
    Loads paired (ISIC_<id>.jpg, ISIC_<id>_segmentation.png).
    """

    def __init__(
        self,
        samples: List[Dict[str, str]],
        transform: Optional[A.Compose] = None,
        img_size: int = 256,
    ):
        self.samples = samples
        self.transform = transform
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image_path = item["image_path"]
        mask_path = item["mask_path"]
        image_id = item["image_id"]

        # Load image (RGB)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (Grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask at {mask_path}")

        # Binarize mask (0 or 255)
        mask = (mask > 127).astype(np.uint8) * 255

        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented["image"]  # (3, H, W)
            mask_transformed = augmented["mask"]  # (H, W)
            # Ensure float binary mask with channel dim (1, H, W)
            if isinstance(mask_transformed, torch.Tensor):
                mask_tensor = (mask_transformed > 127).float().unsqueeze(0)
            else:
                mask_tensor = torch.from_numpy((mask_transformed > 127).astype(np.float32)).unsqueeze(0)
        else:
            # Fallback simple resize and normalization
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            image_norm = (image.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": image_id,
            "image_path": image_path,
            "mask_path": mask_path,
        }
