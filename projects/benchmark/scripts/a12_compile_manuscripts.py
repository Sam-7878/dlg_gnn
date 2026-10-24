#!/usr/bin/env python3
import subprocess
from pathlib import Path

ROOT = Path('/mnt/d/_Work/goat_bank/dlg_gnn')
PAPER_DIR = ROOT / 'projects/benchmark/paper/a12'
MDPI_DIR = PAPER_DIR / 'mdpi'

def compile_latex(work_dir: Path, tex_name: str):
    print(f"\n==========================================")
    print(f"Compiling {tex_name} in {work_dir}")
    print(f"==========================================")
    
    cmd_pass1 = ['pdflatex', '-interaction=nonstopmode', f'{tex_name}.tex']
    cmd_bib   = ['bibtex', f'{tex_name}']
    cmd_pass2 = ['pdflatex', '-interaction=nonstopmode', f'{tex_name}.tex']
    cmd_pass3 = ['pdflatex', '-interaction=nonstopmode', f'{tex_name}.tex']

    print("Pass 1 (pdflatex)...")
    res1 = subprocess.run(cmd_pass1, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("Pass 2 (bibtex)...")
    res_bib = subprocess.run(cmd_bib, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("Pass 3 (pdflatex)...")
    res2 = subprocess.run(cmd_pass2, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("Pass 4 (pdflatex)...")
    res3 = subprocess.run(cmd_pass3, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    log_file = work_dir / f'{tex_name}.log'
    pdf_file = work_dir / f'{tex_name}.pdf'

    if not pdf_file.exists():
        print(f"FAILED: PDF not created: {pdf_file}")
        print("Last output tail:")
        print("\n".join(res3.stdout.splitlines()[-40:]))
        return False

    # Check log for errors or undefined references
    log_content = log_file.read_text(encoding='utf-8', errors='replace')
    undefined_refs = [l for l in log_content.splitlines() if 'Reference' in l and 'undefined' in l]
    undefined_cites = [l for l in log_content.splitlines() if 'Citation' in l and 'undefined' in l]
    overfull_boxes = [l for l in log_content.splitlines() if 'Overfull \\hbox' in l]

    print(f"SUCCESS: Generated {pdf_file.name} (size: {pdf_file.stat().st_size:,} bytes)")
    print(f"Undefined references: {len(undefined_refs)}")
    for r in undefined_refs:
        print(f"  {r}")
    print(f"Undefined citations: {len(undefined_cites)}")
    for c in undefined_cites:
        print(f"  {c}")
    print(f"Overfull \\hbox warnings: {len(overfull_boxes)}")
    if overfull_boxes:
        for b in overfull_boxes[:5]:
            print(f"  {b}")
    return True

if __name__ == '__main__':
    ok1 = compile_latex(PAPER_DIR, 'DLG-Benchmark_A12_Preprints')
    ok2 = compile_latex(MDPI_DIR, 'DLG-Benchmark_A12_AppliedSciences')
    if ok1 and ok2:
        print("\nALL MANUSCRIPTS COMPILED CLEANLY!")
    else:
        print("\nCompilation issues encountered.")
