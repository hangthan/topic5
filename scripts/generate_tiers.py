"""
Generate multiple notebook versions for backup plans.
Tier 1 (Main):  3 seeds × 3 fractions × 1 LR × 25 epochs = 27 runs (~6-7h)
Tier 2 (Safe):  2 seeds × 3 fractions × 1 LR × 20 epochs = 18 runs (~3.5-4.5h)
Tier 3 (Fast):  2 seeds × 2 fractions × 1 LR × 15 epochs = 12 runs (~2-3h)
Tier 4 (Emergency): 1 seed × 1 fraction × 1 LR × 10 epochs = 6 runs (~1h)
"""
import shutil
import re

MAIN_FILE = "kaggle_topic05_segmentation_MAIN.py"

TIERS = {
    "Tier1_Main": {
        "QUICK_RUN": "False",
        "FINAL_EPOCHS": "25",
        "SEEDS": "(42, 2025, 3407)",
        "DATA_FRACTIONS": "(1.0, 0.5, 0.25)",
        "LEARNING_RATES": "(3e-4,)",
        "EARLY_STOP_PATIENCE": "7",
        "runs": 27,
        "est_hours": "6-7",
    },
    "Tier2_Safe": {
        "QUICK_RUN": "False",
        "FINAL_EPOCHS": "20",
        "SEEDS": "(42, 2025)",
        "DATA_FRACTIONS": "(1.0, 0.5, 0.25)",
        "LEARNING_RATES": "(3e-4,)",
        "EARLY_STOP_PATIENCE": "5",
        "runs": 18,
        "est_hours": "3.5-4.5",
    },
    "Tier3_Fast": {
        "QUICK_RUN": "False",
        "FINAL_EPOCHS": "15",
        "SEEDS": "(42, 2025)",
        "DATA_FRACTIONS": "(1.0, 0.25)",
        "LEARNING_RATES": "(3e-4,)",
        "EARLY_STOP_PATIENCE": "5",
        "runs": 12,
        "est_hours": "2-3",
    },
    "Tier4_Emergency": {
        "QUICK_RUN": "False",
        "FINAL_EPOCHS": "10",
        "SEEDS": "(42,)",
        "DATA_FRACTIONS": "(1.0,)",
        "LEARNING_RATES": "(3e-4,)",
        "EARLY_STOP_PATIENCE": "5",
        "runs": 6,
        "est_hours": "0.5-1",
    },
}

# Patterns to replace
PATTERNS = {
    "QUICK_RUN":   r"(QUICK_RUN:\s*bool\s*=\s*)\S+",
    "FINAL_EPOCHS": r"(FINAL_EPOCHS:\s*int\s*=\s*)\S+",
    "SEEDS":       r"(SEEDS:\s*Tuple\[int, \.\.\.\]\s*=\s*)\([^)]+\)",
    "DATA_FRACTIONS": r"(DATA_FRACTIONS:\s*Tuple\[float, \.\.\.\]\s*=\s*)\([^)]+\)",
    "LEARNING_RATES": r"(LEARNING_RATES:\s*Tuple\[float, \.\.\.\]\s*=\s*)\([^)]+\)",
    "EARLY_STOP_PATIENCE": r"(EARLY_STOP_PATIENCE:\s*int\s*=\s*)\S+",
}

with open(MAIN_FILE, 'r', encoding='utf-8') as f:
    base_content = f.read()

for tier_name, config in TIERS.items():
    content = base_content
    
    for key, pattern in PATTERNS.items():
        if key in config:
            val = config[key]
            content = re.sub(pattern, rf"\g<1>{val}", content)
    
    # Write .py
    py_name = f"kaggle_topic05_{tier_name}.py"
    with open(py_name, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Convert to .ipynb using py2ipynb logic
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
    
    nb = new_notebook()
    cells = content.split('# %%')
    for cell in cells:
        cell = cell.strip()
        if not cell:
            continue
        if cell.startswith('[markdown]'):
            lines = cell.split('\n')[1:]
            md_lines = []
            for line in lines:
                if line.startswith('# '):
                    md_lines.append(line[2:])
                elif line.startswith('#'):
                    md_lines.append(line[1:])
                else:
                    md_lines.append(line)
            nb.cells.append(new_markdown_cell('\n'.join(md_lines)))
        else:
            nb.cells.append(new_code_cell(cell))
    
    ipynb_name = f"Topic05_{tier_name}.ipynb"
    with open(ipynb_name, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"✅ {tier_name}:")
    print(f"   .py:    {py_name}")
    print(f"   .ipynb: {ipynb_name}")
    print(f"   Runs:   {config['runs']} | Epochs: {config['FINAL_EPOCHS']} | Est: {config['est_hours']}h")
    print(f"   Seeds:  {config['SEEDS']} | Fractions: {config['DATA_FRACTIONS']}")
    print()

print("=" * 60)
print("SUMMARY:")
print("=" * 60)
for name, cfg in TIERS.items():
    print(f"  {name:20s} | {cfg['runs']:3d} runs | ~{cfg['est_hours']:>5s}h | {cfg['FINAL_EPOCHS']:>2s} epochs | Seeds: {cfg['SEEDS']}")
