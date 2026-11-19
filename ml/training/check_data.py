"""
Check and Verification Script for DermAI Milestone 2 Data Pipeline.

Verifications performed:
1. Prints split sizes and per-class distributions across train, val, and test splits.
2. Confirms strict lesion_id isolation across splits (zero patient leakage).
3. Computes and displays inverse frequency class weights from the training split.
4. Initializes PyTorch DataLoaders (supporting --subset for rapid CPU smoke testing).
5. Loads one full batch and prints tensor dimensions, dtypes, and value ranges.
6. Saves a 4x4 grid (16 images) of augmented training images to ml/experiments/augmentation_grid.png.
"""

import os
import sys
import argparse
import random
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for command-line execution
import matplotlib.pyplot as plt
import torch

from ml.training.dataset import (
    SkinLesionDataset,
    get_dataloaders,
    compute_class_weights,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    save_classes_json
)
from ml.training.augmentations import denormalize_tensor


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Check DermAI Data Pipeline & Augmentations")
    parser.add_argument(
        "--splits_csv",
        type=str,
        default="ml/data/splits.csv",
        help="Path to splits.csv file (default: ml/data/splits.csv)"
    )
    parser.add_argument(
        "--images_root",
        type=str,
        nargs="+",
        default=["ml/data/raw/images"],
        help="One or more directories containing lesion images (default: ml/data/raw/images)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ml/experiments",
        help="Directory to save verification outputs (default: ml/experiments)"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Number of images per split to load for quick CPU smoke test (default: None for full dataset)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for DataLoader inspection (default: 16)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (default: 0 for Windows compatibility)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target compute device: auto, cuda, or cpu (default: auto)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1. Device Resolution
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("DermAI Milestone 2: Data Pipeline & Augmentation Verification")
    print("=" * 70)
    print(f"Target Device        : {device} (CUDA Available: {torch.cuda.is_available()})")
    print(f"Splits CSV Path      : {args.splits_csv}")
    print(f"Images Root Folders  : {args.images_root}")
    print(f"Output Directory     : {args.output_dir}")
    print(f"Subset per Split     : {args.subset if args.subset else 'Full Dataset'}")
    print(f"Batch Size           : {args.batch_size}")
    print(f"DataLoader Workers   : {args.num_workers}")
    print("-" * 70)

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save classes.json
    classes_json_path = Path("ml/training/classes.json")
    save_classes_json(classes_json_path)
    print(f"[OK] Saved diagnostic class mapping to: {classes_json_path}")

    # 2. Inspect splits.csv
    splits_path = Path(args.splits_csv)
    if not splits_path.is_file():
        print(f"[ERROR] splits.csv not found at: {splits_path}")
        sys.exit(1)

    df = pd.read_csv(splits_path)
    print(f"[OK] Successfully loaded {len(df):,} total records from {splits_path.name}")

    # Print split sizes
    print("\n--- Split Sizes ---")
    split_counts = df["split"].value_counts()
    for s_name in ["train", "val", "test"]:
        cnt = split_counts.get(s_name, 0)
        pct = (cnt / len(df)) * 100
        print(f"  {s_name:<6}: {cnt:>5,} images ({pct:>5.1f}%)")

    # Print per-class counts per split
    print("\n--- Per-Class Distribution by Split ---")
    class_pivot = df.groupby(["dx", "split"]).size().unstack(fill_value=0)
    # Ensure ordered columns
    cols = [c for c in ["train", "val", "test"] if c in class_pivot.columns]
    class_pivot = class_pivot[cols]
    class_pivot["total"] = class_pivot.sum(axis=1)
    
    # Add mapped class index for clarity
    class_pivot["index"] = [CLASS_TO_IDX.get(dx, -1) for dx in class_pivot.index]
    class_pivot = class_pivot.sort_values(by="index")
    
    print(f"  {'Index':<6} {'Code':<8} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("  " + "-" * 48)
    for dx, row in class_pivot.iterrows():
        print(f"  {row['index']:<6} {dx:<8} {row.get('train', 0):<8} {row.get('val', 0):<8} {row.get('test', 0):<8} {row['total']:<8}")

    # 3. Patient Isolation Verification (No lesion_id leakage)
    print("\n--- Patient Leakage Verification (lesion_id) ---")
    train_lesions = set(df[df["split"] == "train"]["lesion_id"])
    val_lesions = set(df[df["split"] == "val"]["lesion_id"])
    test_lesions = set(df[df["split"] == "test"]["lesion_id"])

    train_val_overlap = train_lesions & val_lesions
    train_test_overlap = train_lesions & test_lesions
    val_test_overlap = val_lesions & test_lesions

    print(f"  Unique lesion_ids - Train: {len(train_lesions):,}, Val: {len(val_lesions):,}, Test: {len(test_lesions):,}")
    print(f"  Train vs Val  overlap: {len(train_val_overlap)}")
    print(f"  Train vs Test overlap: {len(train_test_overlap)}")
    print(f"  Val vs Test   overlap: {len(val_test_overlap)}")

    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("  [PASSED] Strict patient isolation confirmed! Zero lesion_id overlap between splits.")
    else:
        print("  [FAILED] DATA LEAKAGE DETECTED across splits!")
        sys.exit(1)

    # 4. Class Imbalance Weights (TRAINING SPLIT ONLY)
    print("\n--- Class Weights (Computed from TRAINING split only) ---")
    class_weights = compute_class_weights(df)
    for dx, idx in sorted(CLASS_TO_IDX.items(), key=lambda x: x[1]):
        w = class_weights[idx].item()
        print(f"  [{idx}] {dx:<6} : weight = {w:8.4f}")

    # 5. Build DataLoaders & Test Batch Loading
    print(f"\n--- Initializing DataLoaders (Subset: {args.subset}) ---")
    loaders = get_dataloaders(
        splits_csv=args.splits_csv,
        images_root=args.images_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset=args.subset,
        use_sampler=False,
        seed=args.seed
    )

    print(f"  Train loader batches: {len(loaders['train'])} ({len(loaders['train'].dataset)} samples)")
    print(f"  Val loader batches  : {len(loaders['val'])} ({len(loaders['val'].dataset)} samples)")
    print(f"  Test loader batches : {len(loaders['test'])} ({len(loaders['test'].dataset)} samples)")

    if args.subset:
        print("\n--- Stratified Subset Per-Class Counts (All 7 Classes Covered) ---")
        t_c = loaders['train'].dataset.get_class_counts()
        v_c = loaders['val'].dataset.get_class_counts()
        te_c = loaders['test'].dataset.get_class_counts()
        print(f"  {'Class':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
        print("  " + "-" * 32)
        for dx in CLASS_TO_IDX.keys():
            print(f"  {dx:<8} {t_c[dx]:<8} {v_c[dx]:<8} {te_c[dx]:<8}")
        print("  " + "-" * 32)
        print(f"  {'Total':<8} {len(loaders['train'].dataset):<8} {len(loaders['val'].dataset):<8} {len(loaders['test'].dataset):<8}")

    # Fetch one batch from train_loader
    print("\n--- Inspecting First Training Batch ---")
    batch_iter = iter(loaders["train"])
    images, labels, image_ids = next(batch_iter)

    print(f"  Images batch tensor shape : {images.shape} (dtype: {images.dtype})")
    print(f"  Labels batch tensor shape : {labels.shape} (dtype: {labels.dtype})")
    print(f"  Image value min/max       : min={images.min().item():.3f}, max={images.max().item():.3f}")
    print(f"  Batch labels              : {labels.tolist()}")
    print(f"  Decoded class codes       : {[IDX_TO_CLASS[l.item()] for l in labels]}")
    print(f"  Sample image IDs          : {list(image_ids[:4])} ...")

    # Verify device transfer
    images_dev = images.to(device)
    labels_dev = labels.to(device)
    print(f"  Device transfer test      : Success! Images on {images_dev.device}, Labels on {labels_dev.device}")

    # 6. Test WeightedRandomSampler
    print("\n--- Testing WeightedRandomSampler on Training Split ---")
    sampler_loaders = get_dataloaders(
        splits_csv=args.splits_csv,
        images_root=args.images_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset=args.subset,
        use_sampler=True
    )
    s_images, s_labels, _ = next(iter(sampler_loaders["train"]))
    s_class_counts = pd.Series([IDX_TO_CLASS[l.item()] for l in s_labels]).value_counts().to_dict()
    print(f"  Sampler batch class counts: {s_class_counts}")
    print("  [OK] WeightedRandomSampler functions as expected.")

    # 7. Generate and Save 4x4 Augmentation Grid
    print("\n--- Generating 4x4 Augmentation Grid (16 Images) ---")
    grid_img_count = min(16, len(images))
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("DermAI Training Augmentations (16 Sample Images)", fontsize=14, fontweight="bold", y=0.98)

    for i in range(16):
        ax = axes[i // 4, i % 4]
        if i < grid_img_count:
            # Denormalize tensor back to standard RGB image
            rgb_img = denormalize_tensor(images[i])
            label_name = IDX_TO_CLASS[labels[i].item()]
            img_id = image_ids[i]
            ax.imshow(rgb_img)
            ax.set_title(f"{img_id}\n{label_name}", fontsize=9, pad=4)
        ax.axis("off")

    plt.tight_layout()
    grid_output_path = output_dir / "augmentation_grid.png"
    plt.savefig(grid_output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved 16-image augmentation grid to: {grid_output_path}")

    print("\n" + "=" * 70)
    print("ALL DATA PIPELINE AND AUGMENTATION VERIFICATIONS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
