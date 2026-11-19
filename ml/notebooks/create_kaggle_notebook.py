"""
Script to generate ml/notebooks/02_train_kaggle.ipynb for DermAI Milestone 2.
Updates Kaggle training notebook with corrected dataset mount paths, package structure,
and training cells for EfficientNet-B0, ResNet-50, and Swin Transformer (swin_tiny_patch4_window7_224).
"""

from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "ml" / "notebooks" / "02_train_kaggle.ipynb"

nb = nbf.v4.new_notebook()

nb.metadata = {
    "language_info": {
        "name": "python",
        "version": "3.10"
    },
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "accelerator": "GPU"
}

cells = []

# Cell 1: Header
cells.append(nbf.v4.new_markdown_cell("""# 02. DermAI Milestone 2: Model Training on Kaggle (GPU Acceleration)
### DermAI: Mobile Skin Lesion Screening & Decision Support System

This notebook runs deep learning training and evaluation for **Milestone 2** on Kaggle with a free GPU (NVIDIA T4 or P100).

---

### Workflow Overview:
1. **Verify GPU**: Check CUDA device availability and GPU memory (`nvidia-smi`).
2. **Install Libraries**: Install `timm`, `albumentations`, `segmentation-models-pytorch`, and `pyyaml`.
3. **Environment & Data Setup**:
   - Reconstruct the `ml/` package structure in `/kaggle/working/ml/` from `/kaggle/input/datasets/sivavidhyap/dermai-code/`.
   - Locate HAM10000 images under `/kaggle/input/datasets/kmader/` by scanning with glob.
   - Verify `splits.csv` at `/kaggle/working/ml/data/splits.csv`.
4. **Fast Smoke Test**: Run 1 epoch with `--subset 50` on GPU to verify the complete pipeline.
5. **Full Training (EfficientNet-B0)**:
   - 20 epochs, 3 freeze epochs, batch size 32, lr 0.0001, class-weighted cross-entropy (`ce_weighted`).
   - Output to `/kaggle/working/experiments/efficientnet_b0`.
6. **Package EfficientNet-B0 Output**:
   - Archive `/kaggle/working/experiments` into `efficientnet_b0_output.zip`.
7. **Full Training (ResNet-50)**:
   - 20 epochs, 3 freeze epochs, batch size 32, lr 0.0001, class-weighted cross-entropy (`ce_weighted`).
   - Output to `/kaggle/working/experiments/resnet50`.
8. **Full Training (Swin Transformer)**:
   - Model: `swin_tiny_patch4_window7_224`
   - 20 epochs, 5 freeze epochs, batch size 32, lr 0.0001, class-weighted cross-entropy (`ce_weighted`).
   - `eval_test`: OFF (preserves test split).
   - Output to `/kaggle/working/experiments/swin_tiny_patch4_window7_224`.
9. **Package Experiment Outputs**:
   - Downloadable archive from Kaggle's Output panel.
"""))

# Cell 2: Step 1 Hardware & GPU Check
cells.append(nbf.v4.new_markdown_cell("""## Step 1: Verify Hardware & GPU Acceleration
Ensure Kaggle accelerator is turned on: **Notebook options (right panel) -> Accelerator -> GPU T4 x2 or P100**."""))

cells.append(nbf.v4.new_code_cell("""# Check GPU hardware and PyTorch CUDA support
!nvidia-smi

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")
else:
    print("WARNING: GPU not detected! Make sure you enabled Accelerator -> GPU in the right sidebar.")
"""))

# Cell 3: Step 2 Dependency Installation
cells.append(nbf.v4.new_markdown_cell("""## Step 2: Install Required Libraries"""))

cells.append(nbf.v4.new_code_cell("""!pip install --quiet timm>=0.9.16 albumentations>=1.4.0 segmentation-models-pytorch>=0.3.3 pyyaml>=6.0.1
print("Dependencies installed successfully.")
"""))

# Common setup code block for every training cell
SETUP_CODE_BLOCK = """import sys, os
sys.path.insert(0, "/kaggle/working")
# Copy files into ml/ package structure
os.makedirs("/kaggle/working/ml/training", exist_ok=True)
os.makedirs("/kaggle/working/ml/data", exist_ok=True)
import shutil
src = "/kaggle/input/datasets/sivavidhyap/dermai-code"
if not os.path.exists(src):
    for candidate in [
        "/kaggle/input/datasets/sivavidhyap/dermai-code/dermai-code",
        "/kaggle/input/dermai-code",
        "/kaggle/input/dermai-code/dermai-code",
    ]:
        if os.path.exists(candidate):
            src = candidate
            break

for folder in ["training", "preprocessing", "segmentation", "evaluation"]:
    if os.path.exists(f"{src}/{folder}"):
        shutil.copytree(f"{src}/{folder}", f"/kaggle/working/ml/{folder}", dirs_exist_ok=True)
if os.path.exists(f"{src}/splits.csv"):
    shutil.copy(f"{src}/splits.csv", "/kaggle/working/ml/data/splits.csv")
elif os.path.exists(f"{src}/data/splits.csv"):
    shutil.copy(f"{src}/data/splits.csv", "/kaggle/working/ml/data/splits.csv")
elif os.path.exists(f"{src}/ml/data/splits.csv"):
    shutil.copy(f"{src}/ml/data/splits.csv", "/kaggle/working/ml/data/splits.csv")

for d in ["/kaggle/working/ml", "/kaggle/working/ml/training", "/kaggle/working/ml/data"]:
    open(f"{d}/__init__.py", "w").close()

# Scan HAM10000 images under /kaggle/input/datasets/kmader/
import glob
jpg_files = glob.glob("/kaggle/input/datasets/kmader/**/*.jpg", recursive=True)
if not jpg_files:
    jpg_files = glob.glob("/kaggle/input/**/kmader*/**/*.jpg", recursive=True)
if not jpg_files:
    jpg_files = glob.glob("/kaggle/input/**/*.jpg", recursive=True)

if not jpg_files:
    raise FileNotFoundError("Could not find any .jpg files! Please ensure the HAM10000 dataset is attached.")

first_jpg = jpg_files[0]
image_root = os.path.dirname(first_jpg)
image_dirs = sorted(list(set(os.path.dirname(p) for p in jpg_files)))
images_root_arg = " ".join(image_dirs)
splits_csv_path = "/kaggle/working/ml/data/splits.csv"
"""

# Cell 4: Step 3 Setup Environment and Data Paths
cells.append(nbf.v4.new_markdown_cell("""## Step 3: Setup Python Environment and Data Paths
Reconstructs the `ml/` package structure under `/kaggle/working/ml/` and scans for the HAM10000 image folders under `/kaggle/input/datasets/kmader/`."""))

cells.append(nbf.v4.new_code_cell(f"""{SETUP_CODE_BLOCK}
print(f"Code root copied from: {{src}}")
print(f"Derived image root from first .jpg: {{image_root}}")
print(f"Discovered {{len(image_dirs)}} image directory(ies) containing {{len(jpg_files)}} total images:")
for d in image_dirs:
    print(f"  - {{d}}")
print(f"Splits CSV path: {{splits_csv_path}} (exists: {{os.path.exists(splits_csv_path)}})")
"""))

# Cell 5: Step 4 Fast GPU Smoke Test
cells.append(nbf.v4.new_markdown_cell("""## Step 4: Fast GPU Smoke Test (--subset 50, --epochs 1)
Runs 1 epoch with 50 stratified samples to verify GPU execution, AMP, and metrics in seconds before starting full training."""))

cells.append(nbf.v4.new_code_cell(f"""{SETUP_CODE_BLOCK}
print(">>> Starting 50-sample GPU Smoke Test <<<")
!python -m ml.training.train \\
    --model resnet50 \\
    --images_root {{images_root_arg}} \\
    --splits_csv {{splits_csv_path}} \\
    --output_dir /kaggle/working/experiments/resnet50_smoke \\
    --epochs 1 \\
    --freeze_epochs 1 \\
    --subset 50 \\
    --batch_size 16 \\
    --device auto \\
    --num_workers 2

print("\\n[OK] Smoke test completed successfully!")
"""))

# Cell 6: Step 5 Full Training EfficientNet-B0
cells.append(nbf.v4.new_markdown_cell("""## Step 5: Full Model Training — EfficientNet-B0 (20 Epochs)
Runs full training for EfficientNet-B0 with:
- `--epochs 20 --freeze_epochs 3 --batch_size 32 --lr 0.0001 --loss ce_weighted`
- Output to `/kaggle/working/experiments/efficientnet_b0`"""))

cells.append(nbf.v4.new_code_cell(f"""{SETUP_CODE_BLOCK}
print(">>> Starting Full Training for EfficientNet-B0 (20 Epochs) <<<")
!python -m ml.training.train \\
    --model efficientnet_b0 \\
    --images_root {{images_root_arg}} \\
    --splits_csv {{splits_csv_path}} \\
    --output_dir /kaggle/working/experiments/efficientnet_b0 \\
    --epochs 20 \\
    --freeze_epochs 3 \\
    --batch_size 32 \\
    --lr 0.0001 \\
    --loss ce_weighted \\
    --scheduler cosine \\
    --device auto \\
    --num_workers 2

print("\\n[OK] EfficientNet-B0 training completed successfully!")
"""))

# Cell 7: Step 6 Zip EfficientNet-B0 Outputs
cells.append(nbf.v4.new_markdown_cell("""## Step 6: Package EfficientNet-B0 Outputs
Creates a zip archive of the experiment outputs for download from Kaggle's **Output** panel."""))

cells.append(nbf.v4.new_code_cell("""import shutil
shutil.make_archive("/kaggle/working/efficientnet_b0_output", "zip", "/kaggle/working/experiments")
print("Download efficientnet_b0_output.zip from the Output panel on the right")
"""))

# Cell 8: Step 7 Full Training ResNet-50
cells.append(nbf.v4.new_markdown_cell("""## Step 7: Full Model Training — ResNet-50 (20 Epochs)
Runs full training for ResNet-50 with:
- `--epochs 20 --freeze_epochs 3 --batch_size 32 --lr 0.0001 --loss ce_weighted`
- Output to `/kaggle/working/experiments/resnet50`"""))

cells.append(nbf.v4.new_code_cell(f"""{SETUP_CODE_BLOCK}
print(">>> Starting Full Training for ResNet-50 (20 Epochs) <<<")
!python -m ml.training.train \\
    --model resnet50 \\
    --images_root {{images_root_arg}} \\
    --splits_csv {{splits_csv_path}} \\
    --output_dir /kaggle/working/experiments/resnet50 \\
    --epochs 20 \\
    --freeze_epochs 3 \\
    --batch_size 32 \\
    --lr 0.0001 \\
    --loss ce_weighted \\
    --scheduler cosine \\
    --device auto \\
    --num_workers 2

print("\\n[OK] ResNet-50 training completed successfully!")
"""))

# Cell 9: Step 8 Full Training Swin Transformer
cells.append(nbf.v4.new_markdown_cell("""## Step 8: Full Model Training — Swin Transformer (swin_tiny_patch4_window7_224) (20 Epochs)
Runs full training for Swin Transformer with:
- `--model swin_tiny_patch4_window7_224`
- `--epochs 20 --freeze_epochs 5 --batch_size 32 --lr 0.0001 --loss ce_weighted`
- `eval_test`: OFF (preserves test split)
- Output to `/kaggle/working/experiments/swin_tiny_patch4_window7_224`"""))

cells.append(nbf.v4.new_code_cell(f"""{SETUP_CODE_BLOCK}
print(">>> Starting Full Training for Swin Transformer (swin_tiny_patch4_window7_224) (20 Epochs) <<<")
!python -m ml.training.train \\
    --model swin_tiny_patch4_window7_224 \\
    --images_root {{images_root_arg}} \\
    --splits_csv {{splits_csv_path}} \\
    --output_dir /kaggle/working/experiments/swin_tiny_patch4_window7_224 \\
    --epochs 20 \\
    --freeze_epochs 5 \\
    --batch_size 32 \\
    --lr 0.0001 \\
    --loss ce_weighted \\
    --scheduler cosine \\
    --device auto \\
    --num_workers 2

print("\\n[OK] Swin Transformer training completed successfully!")
"""))

# Cell 10: Step 9 Zip All Outputs
cells.append(nbf.v4.new_markdown_cell("""## Step 9: Package All Model Experiment Outputs
Creates a zip archive of all experiment outputs for download from Kaggle's **Output** panel."""))

cells.append(nbf.v4.new_code_cell("""import shutil
shutil.make_archive("/kaggle/working/all_experiments_output", "zip", "/kaggle/working/experiments")
print("Download all_experiments_output.zip from the Output panel on the right")
"""))

nb.cells = cells

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook successfully written to {NOTEBOOK_PATH}")
