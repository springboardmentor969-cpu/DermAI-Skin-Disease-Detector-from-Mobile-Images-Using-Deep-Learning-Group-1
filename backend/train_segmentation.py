from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import PROJECT_ROOT
from app.ml.segmentation import evaluate_segmentation_checkpoint, train_segmentation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an independent lesion segmentation model on ISBI 2016."
    )
    parser.add_argument("--dataset-directory", type=Path, required=True)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "backend" / "artifacts" / "segmentation",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-attention", action="store_true")
    parser.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        help="Evaluate an existing checkpoint without training.",
    )
    args = parser.parse_args()
    if args.evaluate_checkpoint is not None:
        report = evaluate_segmentation_checkpoint(
            args.dataset_directory.resolve(),
            args.evaluate_checkpoint.resolve(),
            args.output_directory.resolve(),
            seed=args.seed,
            batch_size=args.batch_size,
        )
    else:
        report = train_segmentation(
            args.dataset_directory.resolve(),
            args.output_directory.resolve(),
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            evaluate_attention=args.evaluate_attention,
        )
    print(f"Selected segmentation model: {report['selected_model']}")
    print(f"Metrics: {args.output_directory.resolve() / 'segmentation_report.json'}")


if __name__ == "__main__":
    main()
