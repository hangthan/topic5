import json
with open('Topic05_EncoderDecoder_Segmentation_Kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells'][:3]):
    print(f"Cell {i} Type: {cell['cell_type']}")
    print("Source:")
    for line in cell['source'][:5]:
        print(line.encode('utf-8', 'replace').decode('utf-8', 'replace').strip())
    print("-" * 40)
