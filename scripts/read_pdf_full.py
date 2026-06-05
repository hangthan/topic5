"""Extract full PDF text."""
import fitz

doc = fitz.open("02VF-Final-projects.pdf")
for i, page in enumerate(doc):
    text = page.get_text()
    print(f"--- PAGE {i+1} ---")
    print(text)
doc.close()
