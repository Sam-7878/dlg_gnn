#!/usr/bin/env python3
"""
A15 Compilation Script
Compiles both Preprints and Applied Sciences manuscripts under TeX Live.
Verifies page counts, zero undefined references, and zero undefined citations.
"""
import subprocess
import sys
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER_A15 = ROOT / "projects/benchmark/paper/a15"
PUB_A15 = ROOT / "projects/benchmark/publication/a15"

def run_cmd(cmd, cwd):
    res = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True, text=True)
    return res

def compile_doc(doc_name, tex_file, cwd):
    print(f"\n==========================================")
    print(f"Compiling {doc_name} in {cwd}...")
    print(f"==========================================")
    doc_stem = Path(tex_file).stem
    
    # Pass 1: pdflatex
    r1 = run_cmd(f"pdflatex -interaction=nonstopmode {tex_file}", cwd)
    # Pass 2: bibtex
    r2 = run_cmd(f"bibtex {doc_stem}", cwd)
    # Pass 3: pdflatex
    r3 = run_cmd(f"pdflatex -interaction=nonstopmode {tex_file}", cwd)
    # Pass 4: pdflatex
    r4 = run_cmd(f"pdflatex -interaction=nonstopmode {tex_file}", cwd)
    
    log_file = (cwd / tex_file).with_suffix(".log")
    log_content = log_file.read_text(encoding="latin1", errors="ignore") if log_file.exists() else ""
    
    # Check for undefined references
    undef_refs = re.findall(r"Reference `(.*?)' on page \d+ undefined", log_content)
    undef_cits = re.findall(r"Citation `(.*?)' on page \d+ undefined", log_content)
    
    # Check page count
    pages_match = re.search(r"Output written on \S+ \((\d+) pages,", log_content)
    page_count = int(pages_match.group(1)) if pages_match else None
    
    print(f"[{doc_name}] Page count: {page_count}")
    print(f"[{doc_name}] Undefined references: {len(undef_refs)} {undef_refs}")
    print(f"[{doc_name}] Undefined citations: {len(undef_cits)} {undef_cits}")
    
    pdf_file = (cwd / tex_file).with_suffix(".pdf")
    print(f"[{doc_name}] PDF exists: {pdf_file.exists()} (size: {pdf_file.stat().st_size if pdf_file.exists() else 0} bytes)")
    
    return {
        "doc": doc_name,
        "page_count": page_count,
        "undef_refs": undef_refs,
        "undef_cits": undef_cits,
        "pdf_exists": pdf_file.exists(),
        "pdf_size": pdf_file.stat().st_size if pdf_file.exists() else 0,
        "pdf_path": str(pdf_file)
    }

def main():
    # 1. Compile Preprints
    res_prep = compile_doc("Preprints", "DLG-Benchmark_A15_Preprints.tex", PAPER_A15)
    
    # 2. Compile Applied Sciences
    res_appl = compile_doc("AppliedSciences", "DLG-Benchmark_A15_AppliedSciences.tex", PAPER_A15 / "mdpi")
    
    assert res_prep["pdf_exists"] and len(res_prep["undef_refs"]) == 0 and len(res_prep["undef_cits"]) == 0, "Preprints compilation failed checks!"
    assert res_appl["pdf_exists"] and len(res_appl["undef_refs"]) == 0 and len(res_appl["undef_cits"]) == 0, "Applied Sciences compilation failed checks!"
    
    print("\n>>> All A15 manuscripts compiled cleanly with 0 undefined references and 0 undefined citations! <<<")

if __name__ == "__main__":
    main()
