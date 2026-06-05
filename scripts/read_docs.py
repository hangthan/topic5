"""Extract text from PDF and DOCX files."""
import fitz  # pymupdf

# Read PDF
doc = fitz.open("02VF-Final-projects.pdf")
print("=" * 80)
print("02VF-Final-projects.pdf")
print("=" * 80)
for i, page in enumerate(doc):
    text = page.get_text()
    print(f"\n--- PAGE {i+1} ---")
    print(text)
doc.close()
