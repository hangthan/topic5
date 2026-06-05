"""Compare two .ipynb files: cell types, cell counts, and config differences."""
import json
import sys

def load_nb(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_cell_summary(nb):
    cells = nb['cells']
    code_cells = [c for c in cells if c['cell_type'] == 'code']
    md_cells = [c for c in cells if c['cell_type'] == 'markdown']
    return {
        'total': len(cells),
        'code': len(code_cells),
        'markdown': len(md_cells),
        'code_cells': code_cells,
        'md_cells': md_cells,
        'all_cells': cells,
    }

def extract_config_from_code(cells):
    """Find config-related lines in code cells."""
    config_lines = {}
    keywords = ['QUICK_RUN', 'FINAL_EPOCHS', 'SEEDS', 'DATA_FRACTIONS', 
                'LEARNING_RATES', 'EARLY_STOP_PATIENCE', 'QUICK_EPOCHS',
                'IMG_SIZE', 'BATCH_SIZE', 'MODELS', 'USE_AMP', 'SCHEDULER',
                'WEIGHT_DECAY', 'DICE_WEIGHT']
    for cell in cells:
        src = ''.join(cell['source'])
        for kw in keywords:
            for line in src.split('\n'):
                stripped = line.strip()
                if stripped.startswith(kw) and ('=' in stripped or ':' in stripped):
                    config_lines[kw] = stripped
    return config_lines

# Load both
nb_tier1 = load_nb('topic05-encoderdecoder-segmentation-kaggle.ipynb')
nb_paa = load_nb('topic05-encoderdecoder_PA_A.ipynb')

s1 = get_cell_summary(nb_tier1)
s2 = get_cell_summary(nb_paa)

print("=" * 80)
print("CELL COUNT COMPARISON")
print("=" * 80)
print(f"{'':30s} | {'Old Kaggle':>15s} | {'PA_A (Kaggle)':>15s}")
print("-" * 80)
print(f"{'Total cells':30s} | {s1['total']:>15d} | {s2['total']:>15d}")
print(f"{'Code cells':30s} | {s1['code']:>15d} | {s2['code']:>15d}")
print(f"{'Markdown cells':30s} | {s1['markdown']:>15d} | {s2['markdown']:>15d}")

# Compare configs
cfg1 = extract_config_from_code(s1['code_cells'])
cfg2 = extract_config_from_code(s2['code_cells'])

print("\n" + "=" * 80)
print("CONFIG COMPARISON")
print("=" * 80)
all_keys = sorted(set(list(cfg1.keys()) + list(cfg2.keys())))
for k in all_keys:
    v1 = cfg1.get(k, '(not found)')
    v2 = cfg2.get(k, '(not found)')
    marker = " <<<< DIFFERENT" if v1 != v2 else ""
    print(f"\n  {k}:")
    print(f"    Tier1:  {v1}")
    print(f"    PA_A:   {v2}{marker}")

# Compare code content cell by cell
print("\n" + "=" * 80)
print("CELL-BY-CELL CODE DIFF")
print("=" * 80)

max_cells = max(s1['total'], s2['total'])
diffs_found = 0
for i in range(min(s1['total'], s2['total'])):
    c1 = s1['all_cells'][i]
    c2 = s2['all_cells'][i]
    
    src1 = ''.join(c1['source'])
    src2 = ''.join(c2['source'])
    
    if c1['cell_type'] != c2['cell_type']:
        print(f"\n  Cell {i}: TYPE MISMATCH - Tier1={c1['cell_type']}, PA_A={c2['cell_type']}")
        diffs_found += 1
    elif src1 != src2:
        # Find specific line differences
        lines1 = src1.split('\n')
        lines2 = src2.split('\n')
        diff_lines = []
        for j, (l1, l2) in enumerate(zip(lines1, lines2)):
            if l1 != l2:
                diff_lines.append((j+1, l1.strip(), l2.strip()))
        if len(lines1) != len(lines2):
            diff_lines.append(('LEN', f'{len(lines1)} lines', f'{len(lines2)} lines'))
        
        if diff_lines:
            diffs_found += 1
            print(f"\n  Cell {i} ({c1['cell_type']}): {len(diff_lines)} line(s) differ")
            for line_no, v1, v2 in diff_lines[:5]:
                print(f"    Line {line_no}:")
                print(f"      Tier1: {v1[:100]}")
                print(f"      PA_A:  {v2[:100]}")

if s1['total'] != s2['total']:
    print(f"\n  Cell count differs: Tier1 has {s1['total']}, PA_A has {s2['total']}")
    diffs_found += 1

print(f"\n  Total cells with differences: {diffs_found}")

# File sizes
import os
size1 = os.path.getsize('Topic05_Tier1_Main.ipynb')
size2 = os.path.getsize('topic05-encoderdecoder_PA_A.ipynb')
print(f"\n{'=' * 80}")
print("FILE SIZE")
print(f"{'=' * 80}")
print(f"  Tier1_Main.ipynb:  {size1/1024:.0f} KB")
print(f"  PA_A.ipynb:        {size2/1024:.0f} KB")
