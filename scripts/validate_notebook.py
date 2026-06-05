"""Validate notebook JSON structure."""
import json

NB_PATH = "Topic05_EncoderDecoder_Segmentation_Kaggle.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("nbformat:", nb["nbformat"], ".", nb["nbformat_minor"])
print("kernel:", nb["metadata"]["kernelspec"]["display_name"])

kaggle_meta = nb["metadata"].get("kaggle", {})
print("kaggle.accelerator:", kaggle_meta.get("accelerator", "NOT SET"))
print("kaggle.isInternetEnabled:", kaggle_meta.get("isInternetEnabled", "NOT SET"))

all_ok = True
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] not in ("code", "markdown", "raw"):
        print(f"Cell {i}: INVALID type {c['cell_type']}")
        all_ok = False
    if "source" not in c:
        print(f"Cell {i}: MISSING source")
        all_ok = False
    if c["cell_type"] == "code" and "outputs" not in c:
        print(f"Cell {i}: MISSING outputs")
        all_ok = False

empty_code = [i for i, c in enumerate(nb["cells"]) 
              if c["cell_type"] == "code" and not "".join(c["source"]).strip()]
if empty_code:
    print(f"Empty code cells: {empty_code}")
    all_ok = False

total_code = sum(len(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
total_md = sum(len(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
print(f"Code lines: {total_code}")
print(f"Markdown lines: {total_md}")
print(f"Total: {total_code + total_md}")

# Check QUICK_RUN setting
all_code = ""
for c in nb["cells"]:
    if c["cell_type"] == "code":
        all_code += "".join(c["source"])

if "QUICK_RUN: bool = False" in all_code:
    print("QUICK_RUN: False (full training)")
elif "QUICK_RUN: bool = True" in all_code:
    print("QUICK_RUN: True (debug mode)")

# Check Kvasir auto-scan
if "kaggle_input" in all_code and "glob" in all_code:
    print("Kvasir auto-scan: YES (robust)")
else:
    print("Kvasir auto-scan: NO (hardcoded)")

print()
if all_ok:
    print("=== NOTEBOOK VALID - READY FOR KAGGLE ===")
else:
    print("=== HAS ISSUES - FIX BEFORE UPLOAD ===")
