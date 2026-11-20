from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import PROJECT_ROOT
from app.ml.classification import MODEL_NAMES
from app.ml.training import evaluate_checkpoint, train_and_evaluate, verify_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and compare MobileNetV2 and EfficientNetB0 on HAM10000."
    )
    parser.add_argument("--epochs", type=int, default=20, choices=range(1, 21))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--architecture", choices=MODEL_NAMES, default=None)
    parser.add_argument(
        "--weighting",
        choices=("inverse_frequency", "mild", "weighted_sampler"),
        default="inverse_frequency",
    )
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--no-test-evaluation", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--evaluate-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Directory containing HAM10000_metadata.csv and both image folders.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=False,
        help="New, empty directory for this run's evaluation reports.",
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        required=False,
        help="New, empty directory for this run's model weights and metadata.",
    )
    args = parser.parse_args()
    if args.verify_only:
        if args.split_manifest is None:
            parser.error("--verify-only requires --split-manifest")
        counts = verify_split_manifest(
            args.project_root.resolve(), args.split_manifest.resolve(), args.seed
        )
        print(
            f"Verified configuration: architecture={args.architecture or 'default'} "
            f"weighting={args.weighting} pretrained={not args.no_pretrained} "
            f"seed={args.seed}"
        )
        print(f"Verified split manifest: {counts}")
        return
    if args.evaluate_checkpoint:
        if args.checkpoint_directory is None or args.output_directory is None:
            parser.error("--evaluate-checkpoint requires checkpoint and output directories")
        if args.split_manifest is None:
            parser.error("--evaluate-checkpoint requires --split-manifest")
        report = evaluate_checkpoint(
            args.project_root.resolve(),
            args.checkpoint_directory.resolve(),
            args.output_directory.resolve(),
            args.split_manifest.resolve(),
            args.seed,
        )
        print(f"Evaluated checkpoint: {report['checkpoint']}")
        return
    if args.architecture is None and args.weighting != "inverse_frequency":
        parser.error("--weighting experiments require --architecture")
    if args.split_manifest is None and (
        args.weighting != "inverse_frequency" or args.no_test_evaluation
    ):
        parser.error("This experiment requires --split-manifest to protect the approved split.")
    if args.output_directory is None or args.model_directory is None:
        parser.error("Training requires separate output and model directories.")
    report = train_and_evaluate(
        project_root=args.project_root.resolve(),
        output_directory=args.output_directory.resolve(),
        model_directory=args.model_directory.resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        pretrained=not args.no_pretrained,
        patience=args.patience,
        min_delta=args.min_delta,
        architecture_names=(args.architecture,) if args.architecture else MODEL_NAMES,
        weighting=args.weighting,
        split_manifest=args.split_manifest.resolve(),
        evaluate_test=not args.no_test_evaluation,
    )
    print(f"Selected model: {report['selected_model']}")
    print(f"Evaluation report: {args.output_directory / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()
