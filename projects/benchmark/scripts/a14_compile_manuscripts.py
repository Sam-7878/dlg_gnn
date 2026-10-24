#!/usr/bin/env python3
"""
Compile A14 LaTeX Manuscripts: Preprints.org and MDPI Applied Sciences.
Checks for 0 undefined references, 0 undefined citations, and records page counts.
Repository-relative and portable.
"""

import subprocess
import sys
import re
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PAPER_A14 = BASE_DIR / 'paper' / 'a14'
MDPI_A14 = PAPER_A14 / 'mdpi'
PUB_A14 = BASE_DIR / 'publication' / 'a14'
PUB_A14.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd, cwd):
    print(f"Running: {' '.join(cmd)} in {cwd}")
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return res.returncode, res.stdout

def check_log(log_text):
    undef_refs = re.findall(r"LaTeX Warning: Reference `([^']+)' on page .* undefined", log_text)
    undef_cits = re.findall(r"LaTeX Warning: Citation `([^']+)' on page .* undefined", log_text)
    bib_undef = re.findall(r"Warning--I didn't find a database entry for \"([^\"]+)\"", log_text)
    undef_cits = list(set(undef_cits + bib_undef))
    pages = None
    m = re.search(r"Output written on .* \((\d+) pages,", log_text)
    if m:
        pages = int(m.group(1))
    return undef_refs, undef_cits, pages

def compile_document(doc_name, work_dir, log_out_name):
    print(f"\n==========================================")
    print(f"Compiling {doc_name} in {work_dir}")
    print(f"==========================================")
    
    rc, out1 = run_cmd(["pdflatex", "-interaction=nonstopmode", f"{doc_name}.tex"], work_dir)
    rc, out_bib = run_cmd(["bibtex", doc_name], work_dir)
    rc, out2 = run_cmd(["pdflatex", "-interaction=nonstopmode", f"{doc_name}.tex"], work_dir)
    rc, out3 = run_cmd(["pdflatex", "-interaction=nonstopmode", f"{doc_name}.tex"], work_dir)
    
    full_log = f"=== BIBTEX OUTPUT ===\n{out_bib}\n=== FINAL PDFLATEX OUTPUT ===\n{out3}"
    (PUB_A14 / log_out_name).write_text(full_log, encoding='utf-8')
    
    pdf_path = work_dir / f"{doc_name}.pdf"
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} was not created!")
        print(out3[-2000:])
        return False, [], [], None
    
    undef_refs, undef_cits, pages = check_log(out_bib + "\n" + out3)
    
    print(f"Result for {doc_name}:")
    print(f"  PDF created: {pdf_path.exists()} (size: {pdf_path.stat().st_size:,} bytes)")
    print(f"  Pages: {pages}")
    print(f"  Undefined references: {len(undef_refs)} {undef_refs if undef_refs else ''}")
    print(f"  Undefined citations: {len(undef_cits)} {undef_cits if undef_cits else ''}")
    
    return True, undef_refs, undef_cits, pages

def main():
    ok1, refs1, cits1, pages1 = compile_document("DLG-Benchmark_A14_Preprints", PAPER_A14, "compilation_log_preprints.txt")
    ok2, refs2, cits2, pages2 = compile_document("DLG-Benchmark_A14_AppliedSciences", MDPI_A14, "compilation_log_applsci.txt")
    
    summary = {
        "preprints": {
            "success": ok1,
            "pages": pages1,
            "undef_refs": refs1,
            "undef_cits": cits1
        },
        "applied_sciences": {
            "success": ok2,
            "pages": pages2,
            "undef_refs": refs2,
            "undef_cits": cits2
        }
    }
    
    (PUB_A14 / "compilation_summary.json").write_text(json.dumps(summary, indent=2), encoding='utf-8')
    
    if not (ok1 and ok2):
        print("FAIL: One or both compilations failed.")
        sys.exit(1)
        
    if refs1 or cits1 or refs2 or cits2:
        print("WARNING: Undefined references or citations exist!")
        sys.exit(1)
    else:
        print("\nSUCCESS: Both A14 documents compiled cleanly with 0 undefined references and 0 undefined citations!")

if __name__ == "__main__":
    main()
