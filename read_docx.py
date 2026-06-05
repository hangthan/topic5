"""Extract DOCX content."""
from zipfile import ZipFile
from xml.etree import ElementTree as ET

def read_docx(path):
    with ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = root.findall(".//w:p", ns)
    text_parts = []
    for p in paragraphs:
        runs = p.findall(".//w:r/w:t", ns)
        line = "".join(r.text or "" for r in runs)
        if line.strip():
            text_parts.append(line)
    return "\n".join(text_parts)

# Read Writing Guidelines
print("=" * 80)
print("Writing_guidelines_final_project_report.docx")
print("=" * 80)
text = read_docx("Report-templates-extracted/Writing_guidelines_final_project_report.docx")
print(text)

print("\n\n")

# Read Template
print("=" * 80)
print("Template_final_project_report.docx")
print("=" * 80)
text2 = read_docx("Report-templates-extracted/Template_final_project_report.docx")
print(text2)
