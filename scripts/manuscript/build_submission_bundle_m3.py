#!/usr/bin/env python3
"""
build_submission_bundle_m3.py

Phase G: Builds self-contained submission bundle:
outputs/benchmark/manuscript_m3/submission/DLG_Benchmark_MDPI_Submission.zip

Contains:
- DLG-Benchmark.tex
- references.bib
- Definitions/ (MDPI class, logos, art)
- generated/ (All generated LaTeX tables)
- soul.sty (if present)

Executes clean unpacking and full 4-pass LaTeX compilation:
pdflatex -> bibtex -> pdflatex -> pdflatex
Asserts: 0 errors, 0 missing citations, 0 missing figures.
Copies resulting PDF to:
- outputs/benchmark/manuscript_m3/submission/DLG-Benchmark.pdf
- outputs/benchmark/manuscript_m3/manuscript/DLG-Benchmark.pdf
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_submission_m3")

REPO_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = REPO_ROOT / "docs" / "papers" / "_42_Benchmark"
M3_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m3"
SUBMISSION_DIR = M3_DIR / "submission"
MANUSCRIPT_OUT_DIR = M3_DIR / "manuscript"
ZIP_PATH = SUBMISSION_DIR / "DLG_Benchmark_MDPI_Submission.zip"


def create_submission_zip():
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Creating submission bundle at {ZIP_PATH}...")

    tex_file = MANUSCRIPT_DIR / "DLG-Benchmark.tex"
    bib_file = MANUSCRIPT_DIR / "references.bib"
    definitions_dir = MANUSCRIPT_DIR / "Definitions"
    generated_dir = MANUSCRIPT_DIR / "generated"
    soul_file = MANUSCRIPT_DIR / "soul.sty"

    assert tex_file.exists(), f"Missing {tex_file}"
    assert bib_file.exists(), f"Missing {bib_file}"
    assert definitions_dir.exists(), f"Missing {definitions_dir}"
    assert generated_dir.exists(), f"Missing {generated_dir}"

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Root level files
        zf.write(tex_file, "DLG-Benchmark.tex")
        zf.write(bib_file, "references.bib")
        if soul_file.exists():
            zf.write(soul_file, "soul.sty")

        # Definitions directory
        for f in definitions_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(MANUSCRIPT_DIR)
                zf.write(f, rel.as_posix())

        # Generated directory
        for f in generated_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(MANUSCRIPT_DIR)
                zf.write(f, rel.as_posix())

    zip_size_kb = ZIP_PATH.stat().st_size / 1024
    log.info(f"Created submission bundle: {ZIP_PATH} ({zip_size_kb:.1f} KB)")


def test_compile_bundle():
    log.info("Testing compilation of unpacked submission bundle in clean tempdir...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(tmp_path)

        def run_tex(cmd: list[str], check_error=True):
            res = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
            if check_error and res.returncode != 0:
                log.error(f"Command failed: {' '.join(cmd)}\nStdout:\n{res.stdout}\nStderr:\n{res.stderr}")
                raise RuntimeError(f"LaTeX compile step failed: {cmd[0]}")
            return res

        log.info("Pass 1: pdflatex (initial)...")
        run_tex(["pdflatex", "-interaction=nonstopmode", "DLG-Benchmark.tex"], check_error=False)

        log.info("Pass 2: bibtex...")
        run_tex(["bibtex", "DLG-Benchmark"], check_error=True)

        log.info("Pass 3: pdflatex (citations)...")
        run_tex(["pdflatex", "-interaction=nonstopmode", "DLG-Benchmark.tex"], check_error=False)

        log.info("Pass 4: pdflatex (cross-references & final)...")
        run_tex(["pdflatex", "-interaction=nonstopmode", "DLG-Benchmark.tex"], check_error=False)

        pdf_path = tmp_path / "DLG-Benchmark.pdf"
        log_path = tmp_path / "DLG-Benchmark.log"
        assert pdf_path.exists(), "DLG-Benchmark.pdf was not produced!"
        assert pdf_path.stat().st_size > 50000, f"PDF too small ({pdf_path.stat().st_size} bytes)"

        log_content = log_path.read_text(encoding="utf-8", errors="replace")
        
        # Check for undefined citations
        undef_cites = re.findall(r"Citation `(.*?)' on page .*? undefined", log_content)
        assert len(undef_cites) == 0, f"Found undefined citations in LaTeX compilation: {undef_cites}"

        # Check for fatal LaTeX errors
        fatal_errors = re.findall(r"^! .*", log_content, re.MULTILINE)
        assert len(fatal_errors) == 0, f"Found LaTeX errors: {fatal_errors}"

        # Copy to destination outputs
        dest_pdf1 = SUBMISSION_DIR / "DLG-Benchmark.pdf"
        dest_pdf2 = MANUSCRIPT_OUT_DIR / "DLG-Benchmark.pdf"
        shutil.copy2(pdf_path, dest_pdf1)
        shutil.copy2(pdf_path, dest_pdf2)
        log.info(f"Verified and copied compiled PDF to {dest_pdf1} and {dest_pdf2} ({dest_pdf1.stat().st_size / 1024:.1f} KB)")

    print("Phase G Complete: Self-contained submission bundle compiled cleanly with 0 errors and 0 undefined citations.")


def main():
    create_submission_zip()
    test_compile_bundle()


if __name__ == "__main__":
    main()
