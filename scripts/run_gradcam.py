"""
Run Grad-CAM on a DermAI test image.
"""

from pathlib import Path
import sys


# ============================================================
# ADD DERMAI ROOT TO PYTHON PATH
# ============================================================

DERMAI_ROOT = Path(__file__).resolve().parents[1]

if str(DERMAI_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(DERMAI_ROOT)
    )


# ============================================================
# IMPORT GRAD-CAM
# ============================================================

from xai.gradcam import generate_gradcam


# ============================================================
# CHANGE THIS IMAGE PATH
# ============================================================

IMAGE_PATH = Path(
    r"C:\Users\SRINJOY\Dropbox\PC\Downloads\PAD-UFES-20\images\imgs_part_1\PAT_9_17_80.png"
)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    result = generate_gradcam(
        image_path=IMAGE_PATH
    )

    print()
    print("FINAL RESULT")
    print("-" * 40)

    print(
        f"Condition : {result['condition']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Output    : "
        f"{result['output_path']}"
    )