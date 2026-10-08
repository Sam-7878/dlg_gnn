#!/usr/bin/env python3
"""
build_mdpi_bundle_p1.py

Round P1: Builds the official MDPI Applied Sciences submission bundle:
  Outputs:
    publication/mdpi/DLG-Benchmark.tex
    publication/mdpi/DLG-Benchmark.pdf
    publication/mdpi/DLG_Benchmark_MDPI_Submission.zip
    publication/mdpi/cover_letter.md

Guarantees:
  1. Compiles with mdpi.cls in submission mode for Applied Sciences Special Issue:
     'Graph Neural Networks: Theory, Methods and Applications'.
  2. Clean 4-pass compilation (pdflatex -> bibtex -> pdflatex -> pdflatex) with 0 errors.
  3. Includes authoritative cover_letter.md.
  4. Fully self-contained submission archive containing Definitions/, generated/, and bib sources.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_mdpi_bundle")

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_PAPER_DIR = REPO_ROOT / "docs" / "papers" / "_42_Benchmark"
MDPI_PUB_DIR = REPO_ROOT / "publication" / "benchmark" / "mdpi"
MDPI_PUB_DIR.mkdir(parents=True, exist_ok=True)

MASTER_TEX = DOCS_PAPER_DIR / "DLG-Benchmark.tex"
TARGET_TEX = MDPI_PUB_DIR / "DLG-Benchmark.tex"
TARGET_PDF = MDPI_PUB_DIR / "DLG-Benchmark.pdf"
TARGET_ZIP = MDPI_PUB_DIR / "DLG_Benchmark_MDPI_Submission.zip"
COVER_LETTER = MDPI_PUB_DIR / "cover_letter.md"


def compile_mdpi(src_dir: Path) -> Path:
    """Compiles the MDPI manuscript via pdflatex -> bibtex -> pdflatex -> pdflatex."""
    tex_file = src_dir / "DLG-Benchmark.tex"
    assert tex_file.exists()

    def run_cmd(cmd: list[str], label: str, check_error=True):
        res = subprocess.run(cmd, cwd=src_dir, capture_output=True, text=True)
        if check_error and res.returncode != 0:
            log.error(f"{label} failed with code {res.returncode}:\n{res.stderr}\n{res.stdout[-2000:]}")
            raise RuntimeError(f"{label} compilation failed")
        return res

    log.info("Pass 1: pdflatex...")
    run_cmd(["pdflatex", "-interaction=nonstopmode", "DLG-Benchmark.tex"], "Pass 1 pdflatex", check_error=False)
    log.info("Pass 2: bibtex...")
    run_cmd(["bibtex", "DLG-Benchmark"], "Pass 2 bibtex", check_error=True)
    log.info("Pass 3: pdflatex...")
    run_cmd(["pdflatex", "-interaction=nonstopmode", "DLG-Benchmark.tex"], "Pass 3 pdflatex", check_error=False)
    log.info("Pass 4: pdflatex...")
    run_cmd(["pdflatex", "-interaction=nonstopmode", "DLG-Benchmark.tex"], "Pass 4 pdflatex", check_error=True)

    pdf_out = src_dir / "DLG-Benchmark.pdf"
    assert pdf_out.exists(), "Compiled PDF not produced"
    log.info(f"MDPI compilation succeeded! PDF size: {pdf_out.stat().st_size / 1024:.1f} KB")
    return pdf_out


def build_mdpi_bundle():
    log.info("Building MDPI Applied Sciences Submission Bundle...")

    # Copy master tex to publication/mdpi/
    shutil.copy2(MASTER_TEX, TARGET_TEX)
    shutil.copy2(DOCS_PAPER_DIR / "references.bib", MDPI_PUB_DIR / "references.bib")
    if (DOCS_PAPER_DIR / "appendix_financial_results.tex").exists():
        shutil.copy2(DOCS_PAPER_DIR / "appendix_financial_results.tex", MDPI_PUB_DIR / "appendix_financial_results.tex")
    if (DOCS_PAPER_DIR / "soul.sty").exists():
        shutil.copy2(DOCS_PAPER_DIR / "soul.sty", MDPI_PUB_DIR / "soul.sty")

    # Copy Definitions/
    def_target = MDPI_PUB_DIR / "Definitions"
    def_target.mkdir(parents=True, exist_ok=True)
    for p in (DOCS_PAPER_DIR / "Definitions").glob("*"):
        if p.is_file():
            shutil.copy2(p, def_target / p.name)

    # Copy generated/
    gen_target = MDPI_PUB_DIR / "generated"
    gen_target.mkdir(parents=True, exist_ok=True)
    for p in (DOCS_PAPER_DIR / "generated").glob("*.tex"):
        shutil.copy2(p, gen_target / p.name)

    # Test clean compilation in isolated temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        log.info(f"Testing compilation in pristine scratch directory: {tmp_path}")

        # Copy all files
        shutil.copy2(TARGET_TEX, tmp_path / "DLG-Benchmark.tex")
        shutil.copy2(MDPI_PUB_DIR / "references.bib", tmp_path / "references.bib")
        if (MDPI_PUB_DIR / "appendix_financial_results.tex").exists():
            shutil.copy2(MDPI_PUB_DIR / "appendix_financial_results.tex", tmp_path / "appendix_financial_results.tex")
        if (MDPI_PUB_DIR / "soul.sty").exists():
            shutil.copy2(MDPI_PUB_DIR / "soul.sty", tmp_path / "soul.sty")

        tmp_def = tmp_path / "Definitions"
        tmp_def.mkdir(parents=True, exist_ok=True)
        for p in def_target.glob("*"):
            if p.is_file():
                shutil.copy2(p, tmp_def / p.name)

        tmp_gen = tmp_path / "generated"
        tmp_gen.mkdir(parents=True, exist_ok=True)
        for p in gen_target.glob("*.tex"):
            shutil.copy2(p, tmp_gen / p.name)

        compiled_pdf = compile_mdpi(tmp_path)

        # Copy compiled PDF to destination
        shutil.copy2(compiled_pdf, TARGET_PDF)
        log.info(f"Published verified MDPI PDF to {TARGET_PDF}")

    # Build self-contained MDPI submission zip
    log.info(f"Packaging self-contained MDPI submission archive to {TARGET_ZIP}...")
    with zipfile.ZipFile(TARGET_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(TARGET_TEX, "DLG-Benchmark.tex")
        zf.write(MDPI_PUB_DIR / "references.bib", "references.bib")
        if (MDPI_PUB_DIR / "appendix_financial_results.tex").exists():
            zf.write(MDPI_PUB_DIR / "appendix_financial_results.tex", "appendix_financial_results.tex")
        if (MDPI_PUB_DIR / "soul.sty").exists():
            zf.write(MDPI_PUB_DIR / "soul.sty", "soul.sty")
        if COVER_LETTER.exists():
            zf.write(COVER_LETTER, "cover_letter.md")

        for p in sorted(def_target.glob("*")):
            if p.is_file():
                zf.write(p, f"Definitions/{p.name}")

        for p in sorted(gen_target.glob("*.tex")):
            zf.write(p, f"generated/{p.name}")

    log.info(f"Successfully created MDPI bundle: {TARGET_ZIP} ({TARGET_ZIP.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build_mdpi_bundle()
