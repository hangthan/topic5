import json

with open('(demo)Topic05_EncoderDecoder_Segmentation_Colab_9plus_KvasirReady.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

cells = data['cells']
with open('notebook_content.txt', 'w', encoding='utf-8') as out:
    for i, c in enumerate(cells):
        cell_type = c['cell_type']
        source = ''.join(c['source'])
        out.write(f"\n{'='*80}\n")
        out.write(f"CELL {i} [{cell_type}]\n")
        out.write(f"{'='*80}\n")
        out.write(source)
        out.write('\n')
        
        # Also print outputs for code cells (just text outputs)
        if 'outputs' in c:
            for o in c['outputs']:
                if 'text' in o:
                    text = ''.join(o['text'])
                    if len(text) > 500:
                        text = text[:500] + '... [TRUNCATED]'
                    out.write(f"\n--- OUTPUT ---\n{text}\n")
                elif 'data' in o and 'text/plain' in o['data']:
                    text = ''.join(o['data']['text/plain'])
                    if len(text) > 500:
                        text = text[:500] + '... [TRUNCATED]'
                    out.write(f"\n--- OUTPUT (data) ---\n{text}\n")

print("Done - wrote notebook_content.txt")
