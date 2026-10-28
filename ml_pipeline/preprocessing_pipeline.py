#!/usr/bin/env python3
'''
Standalone CLI for DermAI Milestone 1 Skin Image Preprocessing Pipeline.
Usage:
  python3 preprocessing_pipeline.py --input ../sample_data/images/sample_02_melanoma_hair_artifact.png --output processed.png
'''
import argparse
import os
import sys
import base64

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from app.services.preprocessing import run_preprocessing_pipeline

def main():
    parser = argparse.ArgumentParser(description="DermAI Milestone 1 Image Preprocessing CLI")
    parser.add_argument("--input", required=True, help="Path to raw mobile skin image")
    parser.add_argument("--output", required=True, help="Path to save preprocessed 224x224 PNG")
    parser.add_argument("--crop-ratio", type=float, default=0.90, help="Center ROI crop fraction")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        raw_bytes = f.read()

    res = run_preprocessing_pipeline(raw_bytes, filename=os.path.basename(args.input), crop_ratio=args.crop_ratio)
    b64_data = res["normalized_output_base64"].split(",", 1)[1]
    with open(args.output, "wb") as f:
        f.write(base64.b64decode(b64_data))

    print("Quality Assessment:", res["quality_metrics"])
    print("Normalization Stats:", res["normalization_stats"])
    print(f"Saved preprocessed output to {args.output}")

if __name__ == "__main__":
    main()
