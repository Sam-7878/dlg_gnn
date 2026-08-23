#!/usr/bin/env python3
"""
Dump all sections related to THEIA from TA51_Final_report_E5.docx
"""
from pathlib import Path
from docx import Document

DOCX_PATH = Path("/mnt/d/_Work/_data/DLG/DARPA-TC-THEIA/Ground_Truth/TA51_Final_report_E5.docx")

doc = Document(str(DOCX_PATH))

print("=== ALL PARAGRAPHS MENTIONING THEIA ===")
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if not text:
        continue
    if "theia" in text.lower():
        style = p.style.name if p.style else ""
        print(f"[{i:4d}] {style:20s}: {text}")
