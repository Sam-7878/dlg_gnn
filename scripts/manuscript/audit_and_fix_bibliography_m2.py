#!/usr/bin/env python3
"""
audit_and_fix_bibliography_m2.py

Audits references.bib against DLG-Benchmark.tex:
1. Validates all cited keys in DLG-Benchmark.tex exist in references.bib.
2. Identifies unused bibliography entries.
3. Fixes metadata for tang2022revisiting and zheng2021generative.
4. Removes invalid gao2024survey entry and updates citation in DLG-Benchmark.tex.
5. Writes reference_metadata_audit.csv to outputs/benchmark/manuscript_m2/bibliography/.
"""

import re
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs" / "papers" / "_42_Benchmark"
BIB_FILE = DOCS_DIR / "references.bib"
TEX_FILE = DOCS_DIR / "DLG-Benchmark.tex"
OUT_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m2" / "bibliography"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORRECT_TANG = """@inproceedings{tang2022revisiting,
  title     = {Rethinking Graph Neural Networks for Anomaly Detection},
  author    = {Tang, Jianheng and Li, Jiajin and Gao, Ziqi and Li, Jia},
  booktitle = {International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {162},
  pages     = {21076--21089},
  year      = {2022},
  publisher = {PMLR}
}"""

CORRECT_ZHENG = """@article{zheng2021generative,
  title   = {Generative Pre-Training for Graph Neural Networks},
  author  = {Zheng, Yu and Jin, Ming and Liu, Yixin and Chi, Lianhua and Phan, K. T. and Chen, Yi-Ping Phoebe},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  volume  = {35},
  number  = {12},
  pages   = {12220--12233},
  year    = {2023},
  doi     = {10.1109/TKDE.2021.3119326}
}"""

def extract_cited_keys(tex_path: Path):
    text = tex_path.read_text(encoding="utf-8")
    # match \cite{key1, key2}
    citations = set()
    for match in re.finditer(r'\\cite[a-zA-Z]*\{([^}]+)\}', text):
        keys = match.group(1).split(',')
        for k in keys:
            k = k.strip()
            if k:
                citations.add(k)
    return citations

def parse_bib_keys_and_blocks(bib_path: Path):
    text = bib_path.read_text(encoding="utf-8")
    # Match entry starting with @something{key,
    entries = {}
    pattern = re.compile(r'@([a-zA-Z]+)\s*\{\s*([a-zA-Z0-9_\-:]+)\s*,', re.MULTILINE)
    
    # We can find all entries by splitting on @
    chunks = text.split('@')
    header = chunks[0]
    
    parsed = []
    for chunk in chunks[1:]:
        chunk = '@' + chunk
        m = pattern.match(chunk)
        if m:
            entry_type = m.group(1).lower()
            key = m.group(2)
            parsed.append((key, entry_type, chunk))
            entries[key] = (entry_type, chunk)
    return header, parsed, entries

def main():
    print("Auditing bibliography...")
    tex_text = TEX_FILE.read_text(encoding="utf-8")
    
    # Step 1: Update DLG-Benchmark.tex if it cites gao2024survey
    if "gao2024survey" in tex_text:
        tex_text = re.sub(r'\\cite\{ma2023survey\s*,\s*gao2024survey\}', r'\\cite{ma2023survey}', tex_text)
        tex_text = re.sub(r'\\cite\{gao2024survey\s*,\s*ma2023survey\}', r'\\cite{ma2023survey}', tex_text)
        TEX_FILE.write_text(tex_text, encoding="utf-8")
        print("Updated DLG-Benchmark.tex to remove citation to gao2024survey.")

    cited_keys = extract_cited_keys(TEX_FILE)
    print(f"Total citations found in manuscript: {len(cited_keys)}")

    header, parsed_entries, entries_dict = parse_bib_keys_and_blocks(BIB_FILE)
    print(f"Total entries in references.bib: {len(parsed_entries)}")

    # Check for missing cited keys
    missing = cited_keys - set(entries_dict.keys())
    if missing:
        print(f"WARNING: Cited keys missing from bib: {missing}")
    else:
        print("All cited keys exist in bibliography.")

    # Reconstruct cleaned bib
    new_entries = []
    audit_rows = []

    for key, entry_type, chunk in parsed_entries:
        is_cited = key in cited_keys
        status = "RETAINED"
        note = ""

        if key == "gao2024survey":
            status = "PRUNED"
            note = "Invalid unresolvable DOI 10.1145/3696452; subsumed by ma2023survey"
            # Don't add to new_entries
        elif key == "tang2022revisiting":
            status = "CORRECTED"
            note = "Corrected ICML 2022 PMLR author list and metadata (Tang, Li, Gao, Li)"
            new_entries.append((key, CORRECT_TANG))
        elif key == "zheng2021generative":
            status = "CORRECTED"
            note = "Corrected TKDE 2023 title, authors, volume, pages, and valid DOI 10.1109/TKDE.2021.3119326"
            new_entries.append((key, CORRECT_ZHENG))
        else:
            if not is_cited:
                status = "UNUSED_RETAINED"
                note = "Not directly cited in current draft"
            new_entries.append((key, chunk.strip()))

        audit_rows.append({
            "citation_key": key,
            "entry_type": entry_type,
            "is_cited": is_cited,
            "status": status,
            "notes": note
        })

    # Write cleaned references.bib
    final_bib = header.strip() + "\n\n" + "\n\n".join([c for _, c in new_entries]) + "\n"
    BIB_FILE.write_text(final_bib, encoding="utf-8")
    print(f"Wrote cleaned references.bib ({len(new_entries)} entries)")

    # Write audit CSV
    audit_csv = OUT_DIR / "reference_metadata_audit.csv"
    with open(audit_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["citation_key", "entry_type", "is_cited", "status", "notes"])
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"Wrote audit CSV to {audit_csv}")

if __name__ == "__main__":
    main()
