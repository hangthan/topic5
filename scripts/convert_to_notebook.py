"""
Script chuyển đổi file .py (với # %% markers) thành Jupyter Notebook .ipynb
để upload lên Kaggle.

Usage:
    python convert_to_notebook.py
"""
import json
import re
import sys

INPUT_FILE = "kaggle_topic05_segmentation.py"
OUTPUT_FILE = "Topic05_EncoderDecoder_Segmentation_Kaggle.ipynb"


def py_to_ipynb(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by cell markers
    # Patterns: "# %%", "# %% [markdown]"
    cell_pattern = re.compile(r'^# %%(?:\s*\[markdown\])?\s*$', re.MULTILINE)
    
    # Find all cell boundaries
    matches = list(cell_pattern.finditer(content))
    
    cells = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        
        marker = match.group().strip()
        is_markdown = "[markdown]" in marker
        
        cell_content = content[start:end].strip()
        
        if not cell_content:
            continue
        
        if is_markdown:
            # Remove leading "# " from each line for markdown cells
            lines = cell_content.split("\n")
            md_lines = []
            for line in lines:
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line.strip() == "#":
                    md_lines.append("")
                else:
                    md_lines.append(line)
            
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [l + "\n" for l in md_lines]
            })
        else:
            # Code cell
            lines = cell_content.split("\n")
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [l + "\n" for l in lines]
            })
    
    # Remove trailing newline from last line of each cell
    for cell in cells:
        if cell["source"] and cell["source"][-1].endswith("\n"):
            cell["source"][-1] = cell["source"][-1].rstrip("\n")
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            },
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [],
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"[OK] Converted: {input_path} -> {output_path}")
    print(f"  Total cells: {len(cells)}")
    print(f"  Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
    print(f"  Code: {sum(1 for c in cells if c['cell_type'] == 'code')}")


if __name__ == "__main__":
    py_to_ipynb(INPUT_FILE, OUTPUT_FILE)
