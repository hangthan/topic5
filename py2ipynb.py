"""Convert Python file with # %% comments to Jupyter Notebook."""
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import sys

def convert(py_file, ipynb_file):
    nb = new_notebook()
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cells = content.split('# %%')
    
    for cell in cells:
        cell = cell.strip()
        if not cell:
            continue
        
        if cell.startswith('[markdown]'):
            # It's a markdown cell
            # Remove [markdown] and the leading '#' from each line
            lines = cell.split('\n')[1:] # Skip [markdown] line
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
            # It's a code cell
            nb.cells.append(new_code_cell(cell))
            
    with open(ipynb_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    convert('kaggle_topic05_segmentation.py', 'Topic05_EncoderDecoder_Segmentation_Kaggle.ipynb')
    print("Successfully converted to Topic05_EncoderDecoder_Segmentation_Kaggle.ipynb")
