"""Review notebook structure and check for issues."""
import json
import ast
import sys

NB_PATH = "Topic05_EncoderDecoder_Segmentation_Kaggle.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Total cells: {len(cells)} (Markdown: {md_count}, Code: {code_count})")
print()

# Show cell structure
for i, c in enumerate(cells):
    src = "".join(c["source"])
    first_line = src.strip().split("\n")[0][:90] if src.strip() else "(empty)"
    lines = len(c["source"])
    print(f"Cell {i:2d} [{c['cell_type']:8s}] {lines:4d}L | {first_line}")

# Syntax check all code cells
print("\n" + "=" * 60)
print("SYNTAX CHECK")
print("=" * 60)
errors = []
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    # Remove shell commands (lines starting with !)
    lines = src.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!"):
            clean_lines.append("# " + line)  # comment out shell commands
        else:
            clean_lines.append(line)
    clean_src = "\n".join(clean_lines)
    
    try:
        ast.parse(clean_src)
        print(f"Cell {i:2d}: OK")
    except SyntaxError as e:
        print(f"Cell {i:2d}: SYNTAX ERROR at line {e.lineno}: {e.msg}")
        print(f"         {e.text}")
        errors.append((i, e))

# Check imports
print("\n" + "=" * 60)
print("IMPORT CHECK")
print("=" * 60)
all_code = ""
for c in cells:
    if c["cell_type"] == "code":
        all_code += "".join(c["source"]) + "\n"

required_imports = [
    "torch", "torch.nn", "torch.nn.functional", 
    "torchvision", "numpy", "pandas", "matplotlib",
    "albumentations", "tqdm", "PIL", "json", "os", "time",
    "pathlib", "dataclasses", "math", "random"
]
for imp in required_imports:
    if imp in all_code:
        print(f"  {imp:30s} OK")
    else:
        print(f"  {imp:30s} MISSING!")

# Check critical functions/classes exist
print("\n" + "=" * 60)
print("CRITICAL DEFINITIONS CHECK")
print("=" * 60)
critical_defs = [
    "class Config", "def detect_environment", "def set_seed", "def get_device",
    "class OxfordPetSegDataset", "class KvasirSEGDataset", 
    "def get_train_augmentation", "def get_val_augmentation",
    "def split_dataset", "def build_dataset", "class _SubsetWithTransform",
    "class DoubleConv", "class UpBlock",
    "class UNetResNet34", "class SegNetVGG16", 
    "class ASPPConv", "class ASPPPooling", "class ASPP", "class DeepLabV3PlusResNet34",
    "def build_model", "def count_parameters",
    "def calculate_metrics", "def dice_loss", "class HybridLoss",
    "def compute_confusion_matrix", "def estimate_boundary_f1",
    "class EarlyStopping", "def train_one_epoch", "def evaluate",
    "def train_model_one_run", "def make_loaders", "def make_fraction_subset",
    "def measure_inference_latency", "def measure_memory_footprint",
    "def run_experiments",
    "def plot_training_curves", "def plot_model_comparison",
    "def visualize_predictions",
]
missing_defs = []
for d in critical_defs:
    if d in all_code:
        print(f"  {d:45s} OK")
    else:
        print(f"  {d:45s} MISSING!")
        missing_defs.append(d)

# Check for common issues
print("\n" + "=" * 60)
print("POTENTIAL ISSUES CHECK")
print("=" * 60)

# Check QUICK_RUN default
if "QUICK_RUN: bool = False" in all_code:
    print("  QUICK_RUN default = False          OK (full training)")
elif "QUICK_RUN: bool = True" in all_code:
    print("  QUICK_RUN default = True           WARNING (debug mode)")

# Check AMP usage
if "torch.amp.autocast" in all_code:
    print("  AMP autocast (new API)             OK")
elif "torch.cuda.amp.autocast" in all_code:
    print("  AMP autocast (old API)             WARNING - may be deprecated")

if "torch.amp.GradScaler" in all_code:
    print("  GradScaler (new API)               OK")
elif "torch.cuda.amp.GradScaler" in all_code:
    print("  GradScaler (old API)               WARNING - may be deprecated")

# Check scheduler
if "CosineAnnealingLR" in all_code:
    print("  CosineAnnealingLR                  OK")

# Check early stopping
if "EarlyStopping" in all_code:
    print("  EarlyStopping                      OK")

# Check IMG_SIZE
if "IMG_SIZE: int = 256" in all_code:
    print("  IMG_SIZE = 256                     OK")

# Check pretrained weights API
if "IMAGENET1K_V1" in all_code:
    print("  Pretrained weights (new API)       OK")
elif "pretrained=True" in all_code:
    print("  Pretrained weights (old API)       WARNING")

# Check for duplicated function
lines = all_code.split("\n")
func_defs = [l.strip() for l in lines if l.strip().startswith("def ") or l.strip().startswith("class ")]
from collections import Counter
duplicates = {k: v for k, v in Counter(func_defs).items() if v > 1}
if duplicates:
    print(f"\n  DUPLICATE DEFINITIONS:")
    for k, v in duplicates.items():
        print(f"    {k} appears {v} times")
else:
    print("  No duplicate definitions           OK")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if errors:
    print(f"  SYNTAX ERRORS: {len(errors)}")
    for i, e in errors:
        print(f"    Cell {i}: {e.msg} (line {e.lineno})")
else:
    print("  Syntax: ALL OK")

if missing_defs:
    print(f"  MISSING DEFINITIONS: {len(missing_defs)}")
    for d in missing_defs:
        print(f"    {d}")
else:
    print("  Definitions: ALL OK")

print(f"\n  Total lines of code: {len(lines)}")
print(f"  Ready for Kaggle: {'YES' if not errors and not missing_defs else 'NO - FIX ISSUES ABOVE'}")
