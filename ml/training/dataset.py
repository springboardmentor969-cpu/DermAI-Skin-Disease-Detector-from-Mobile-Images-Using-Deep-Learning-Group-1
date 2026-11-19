"""
PyTorch Dataset and DataLoader pipeline for DermAI Milestone 2.

Features:
- Reads ml/data/splits.csv (train, val, test).
- Resolves image_id.jpg across multiple input directories (e.g., Kaggle Part 1 & Part 2).
- Returns (image_tensor, label_index, image_id).
- Maps diagnoses to fixed indices (akiec: 0, bcc: 1, bkl: 2, df: 3, mel: 4, nv: 5, vasc: 6).
- Stratified random sampling for --subset N (guarantees all 7 classes appear, at least 2 per class).
- Computes class weights on the FULL TRAINING split only (inverse class frequency).
- Supports WeightedRandomSampler for class imbalance mitigation.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Union, Optional, Dict, Tuple
import pandas as pd
import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.augmentations import get_transforms

# Fixed 7-class diagnostic mapping for HAM10000
CLASS_TO_IDX: Dict[str, int] = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES: int = len(CLASS_TO_IDX)


def save_classes_json(output_path: Union[str, Path] = "ml/training/classes.json") -> None:
    """Saves the official diagnostic class mapping to a JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(CLASS_TO_IDX, f, indent=2)


def get_stratified_subset(
    df: pd.DataFrame,
    subset_size: int,
    seed: int = 42,
    stratify_col: str = "dx",
    min_per_class: int = 2
) -> pd.DataFrame:
    """
    Extracts a stratified random sample of approximately `subset_size` images,
    guaranteeing that every class present in `df` appears (with at least min_per_class
    samples when the class has >= min_per_class samples in the split).
    Never takes simply the first N rows.
    """
    if subset_size is None or subset_size <= 0 or subset_size >= len(df):
        return df.copy().reset_index(drop=True)

    classes = sorted(df[stratify_col].unique())
    total_rows = len(df)

    # 1. Calculate proportional target count per class with minimum constraint
    target_counts: Dict[str, int] = {}
    for c in classes:
        class_total = int((df[stratify_col] == c).sum())
        prop = (class_total / total_rows) * subset_size
        floor_min = min(min_per_class, class_total)
        allocated = max(floor_min, int(round(prop)))
        allocated = min(allocated, class_total)
        target_counts[c] = allocated

    # 2. Reconcile difference with target subset_size
    diff = subset_size - sum(target_counts.values())
    sorted_by_size = sorted(classes, key=lambda c: (df[stratify_col] == c).sum(), reverse=True)

    idx = 0
    while diff > 0 and idx < len(sorted_by_size):
        c = sorted_by_size[idx % len(sorted_by_size)]
        class_total = int((df[stratify_col] == c).sum())
        if target_counts[c] < class_total:
            target_counts[c] += 1
            diff -= 1
        idx += 1

    idx = 0
    while diff < 0 and idx < len(sorted_by_size):
        c = sorted_by_size[idx % len(sorted_by_size)]
        floor_min = min(min_per_class, int((df[stratify_col] == c).sum()))
        if target_counts[c] > floor_min:
            target_counts[c] -= 1
            diff += 1
        idx += 1

    # 3. Deterministically sample indices per class
    rng = np.random.RandomState(seed)
    sampled_indices: List[int] = []
    for c in classes:
        c_indices = df[df[stratify_col] == c].index.to_numpy()
        k = target_counts[c]
        sampled = rng.choice(c_indices, size=k, replace=False)
        sampled_indices.extend(sampled.tolist())

    # Shuffle sampled rows so batches aren't ordered by class
    rng.shuffle(sampled_indices)
    return df.loc[sampled_indices].copy().reset_index(drop=True)


class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for DermAI Skin Lesion classification.
    
    Each item returns a tuple of:
        (image_tensor, label_index, image_id)
    """
    def __init__(
        self,
        splits_csv: Union[str, Path] = "ml/data/splits.csv",
        images_roots: Union[str, Path, List[Union[str, Path]]] = "ml/data/raw/images",
        split: str = "train",
        transform = None,
        subset: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Args:
            splits_csv: Path to splits.csv with columns ['image_id', 'dx', 'split', 'lesion_id', ...]
            images_roots: Directory or list of directories containing the .jpg images.
            split: 'train', 'val', or 'test'.
            transform: Callable transformation pipeline. If None, default pipeline for the split is used.
            subset: If specified, perform stratified random sampling for approximately N images.
            random_seed: Seed for reproducible subset selection.
        """
        self.splits_csv = Path(splits_csv)
        self.split = split.lower().strip()
        self.subset = subset
        self.random_seed = random_seed
        
        # Ensure images_roots is a list of Path objects
        if isinstance(images_roots, (str, Path)):
            self.images_roots = [Path(images_roots)]
        else:
            self.images_roots = [Path(p) for p in images_roots]

        # Read and filter splits dataframe
        if not self.splits_csv.is_file():
            raise FileNotFoundError(f"Splits file not found: {self.splits_csv}")
            
        full_df = pd.read_csv(self.splits_csv)
        if "split" not in full_df.columns or "dx" not in full_df.columns or "image_id" not in full_df.columns:
            raise ValueError(f"splits_csv must contain columns ['image_id', 'dx', 'split']. Found: {list(full_df.columns)}")

        # Filter by requested split
        raw_split_df = full_df[full_df["split"] == self.split].copy().reset_index(drop=True)
        if len(raw_split_df) == 0:
            raise ValueError(f"No records found for split '{self.split}' in {self.splits_csv}")

        # Apply STRATIFIED random subset if requested
        if subset is not None and subset > 0:
            self.df = get_stratified_subset(
                raw_split_df,
                subset_size=subset,
                seed=random_seed,
                stratify_col="dx",
                min_per_class=2
            )
        else:
            self.df = raw_split_df

        # Build transform pipeline if not supplied
        if transform is None:
            self.transform = get_transforms(split=self.split)
        else:
            self.transform = transform

        # Pre-scan image roots for fast O(1) file resolution
        self.image_path_cache: Dict[str, str] = {}
        self._build_image_cache()

    def get_class_counts(self) -> Dict[str, int]:
        """Returns per-class sample counts in canonical CLASS_TO_IDX order."""
        counts = self.df["dx"].value_counts().to_dict()
        return {dx: counts.get(dx, 0) for dx in CLASS_TO_IDX.keys()}

    def _build_image_cache(self) -> None:
        """Indexes available image files across all provided root folders."""
        for root in self.images_roots:
            if not root.is_dir():
                continue
            try:
                for entry in os.scandir(root):
                    if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                        stem = os.path.splitext(entry.name)[0]
                        if stem not in self.image_path_cache:
                            self.image_path_cache[stem] = entry.path
            except PermissionError:
                pass

    def _resolve_image_path(self, image_id: str) -> str:
        """Finds the absolute file path for a given image_id across all search roots."""
        # 1. Check fast cache
        if image_id in self.image_path_cache:
            return self.image_path_cache[image_id]

        # 2. Check explicitly across directories
        for root in self.images_roots:
            for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                candidate = root / f"{image_id}{ext}"
                if candidate.is_file():
                    self.image_path_cache[image_id] = str(candidate)
                    return str(candidate)

        # 3. Raise informative error if not found in any search root
        roots_str = ", ".join(str(r) for r in self.images_roots)
        raise FileNotFoundError(
            f"Image '{image_id}.jpg' could not be found in any of the specified images_root folders: [{roots_str}]. "
            f"Please verify paths or attach required datasets."
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image_id = str(row["image_id"]).strip()
        dx = str(row["dx"]).strip()

        if dx not in CLASS_TO_IDX:
            raise KeyError(f"Unknown diagnosis label '{dx}' for image '{image_id}'. Expected one of {list(CLASS_TO_IDX.keys())}")
        label_idx = CLASS_TO_IDX[dx]

        img_path = self._resolve_image_path(image_id)
        try:
            with PILImage.open(img_path) as img:
                img = img.convert("RGB")
                if self.transform is not None:
                    image_tensor = self.transform(img)
                else:
                    image_tensor = transforms.ToTensor()(img)
        except Exception as e:
            raise RuntimeError(f"Error loading image '{image_id}' from '{img_path}': {e}")

        return image_tensor, label_idx, image_id


# ==============================================================================
# Class Imbalance & Weighting Utilities (FULL TRAINING SPLIT ONLY)
# ==============================================================================

def compute_class_weights(
    splits_csv_or_df: Union[str, Path, pd.DataFrame] = "ml/data/splits.csv",
    num_classes: int = NUM_CLASSES
) -> torch.Tensor:
    """
    Computes inverse-frequency class weights computed from the FULL TRAINING SPLIT ONLY.
    
    Formula:
        weight[c] = total_train_samples / (num_classes * count[c])
        
    Args:
        splits_csv_or_df: Path to splits.csv or DataFrame containing full training records.
        num_classes: Number of distinct classes (default: 7).
        
    Returns:
        torch.FloatTensor of shape (num_classes,) ready for nn.CrossEntropyLoss(weight=weights).
    """
    if isinstance(splits_csv_or_df, (str, Path)):
        df = pd.read_csv(splits_csv_or_df)
        df_train = df[df["split"] == "train"]
    else:
        df = splits_csv_or_df
        if "split" in df.columns:
            df_train = df[df["split"] == "train"]
        else:
            df_train = df

    total_samples = len(df_train)
    counts = df_train["dx"].value_counts().to_dict()

    weights = np.zeros(num_classes, dtype=np.float32)
    for dx, idx in CLASS_TO_IDX.items():
        c_count = counts.get(dx, 0)
        if c_count > 0:
            weights[idx] = total_samples / (num_classes * c_count)
        else:
            weights[idx] = 1.0

    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(
    train_dataset_df: pd.DataFrame,
    class_weights: Optional[torch.Tensor] = None
) -> WeightedRandomSampler:
    """
    Constructs a PyTorch WeightedRandomSampler for a training dataset.
    Uses class weights derived from the training set to sample underrepresented classes uniformly.
    """
    if class_weights is None:
        class_weights = compute_class_weights("ml/data/splits.csv")

    sample_weights = [class_weights[CLASS_TO_IDX[dx]].item() for dx in train_dataset_df["dx"]]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    return sampler


# ==============================================================================
# DataLoaders Factory
# ==============================================================================

def get_dataloaders(
    splits_csv: Union[str, Path] = "ml/data/splits.csv",
    images_root: Union[str, Path, List[Union[str, Path]]] = "ml/data/raw/images",
    batch_size: int = 32,
    num_workers: int = 0,
    subset: Optional[int] = None,
    use_sampler: bool = False,
    img_size: int = 224,
    pin_memory: bool = True,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Creates and returns PyTorch DataLoaders for 'train', 'val', and 'test' splits.
    
    Args:
        splits_csv: Path to splits.csv.
        images_root: Single folder or list of folders containing the image files.
        batch_size: Number of images per batch.
        num_workers: Multiprocessing workers (0 recommended on Windows).
        subset: If provided, extracts stratified random subsets covering all 7 classes.
        use_sampler: If True, uses WeightedRandomSampler on the training split.
        img_size: Target image dimensions (default: 224).
        pin_memory: Speeds up host-to-device transfers (auto-disabled if CPU only).
        seed: Random seed for stratified subset selection.
        
    Returns:
        Dictionary mapping {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}.
    """
    pin_mem = pin_memory and torch.cuda.is_available()

    # Datasets with stratified random sampling if subset is requested
    train_dataset = SkinLesionDataset(
        splits_csv=splits_csv,
        images_roots=images_root,
        split="train",
        subset=subset,
        random_seed=seed,
        transform=get_transforms("train", img_size=img_size)
    )

    val_dataset = SkinLesionDataset(
        splits_csv=splits_csv,
        images_roots=images_root,
        split="val",
        subset=subset,
        random_seed=seed,
        transform=get_transforms("val", img_size=img_size)
    )

    test_dataset = SkinLesionDataset(
        splits_csv=splits_csv,
        images_roots=images_root,
        split="test",
        subset=subset,
        random_seed=seed,
        transform=get_transforms("test", img_size=img_size)
    )

    # Compute full class weights for sampler
    full_class_weights = compute_class_weights(splits_csv)

    if use_sampler:
        train_sampler = get_weighted_sampler(train_dataset.df, class_weights=full_class_weights)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=False
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=False
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
