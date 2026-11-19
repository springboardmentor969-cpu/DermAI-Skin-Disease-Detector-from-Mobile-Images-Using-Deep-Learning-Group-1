"""
Convenience entrypoint to run ml/preprocessing/test_preprocessing.py
"""

from ml.preprocessing.test_preprocessing import run_evaluation, Path, PROJECT_ROOT
import sys

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "ml" / "preprocessing" / "sample_images")
    run_evaluation(Path(folder))
