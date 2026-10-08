#!/usr/bin/env python3
"""
build_submission_bundle_m5.py

Round M5 Submission Bundle Builder:
Outputs:
outputs/benchmark/manuscript_m5/submission/DLG_Benchmark_MDPI_Submission.zip
outputs/benchmark/manuscript_m5/submission/DLG-Benchmark.pdf
outputs/benchmark/manuscript_m5/manuscript/DLG-Benchmark.pdf

Executes clean unpacking and full 4-pass LaTeX compilation:
pdflatex -> bibtex -> pdflatex -> pdflatex
Asserts: 0 errors, 0 missing citations, 0 missing figures, page count >= 29.
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
log = logging.getLogger("build_submission_m5")

REPO_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = REPO_ROOT / "docs" / "papers" / "_42_Benchmark"
M5_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m5"
SUBMISSION_DIR = M5_DIR / "submission"
MANUSCRIPT_OUT_DIR = M5_DIR / "manuscript"
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
    assert (generated_dir / "table_capacity_controls_paired_deltas.tex").exists(), "Missing table_capacity_controls_paired_deltas.tex"

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

        log.info("Pass 4: pdflatex (final cross-references)...")
        res_final = run_tex(["pdflatex", "-interaction=nonstopmode", "DLG-Benchmark.tex"], check_error=True)

        # Check log for errors and undefined citations
        log_content = (tmp_path / "DLG-Benchmark.log").read_text(encoding="utf-8", errors="ignore")
        assert "! LaTeX Error:" not in log_content, "LaTeX Error found in log"
        assert "! Fatal error" not in log_content, "Fatal error found in log"

        undefined_cites = re.findall(r"Citation `(.*?)' on page \d+ undefined", log_content)
        assert len(undefined_cites) == 0, f"Undefined citations found: {undefined_cites}"

        # Copy compiled PDF to submission and manuscript output directories
        compiled_pdf = tmp_path / "DLG-Benchmark.pdf"
        assert compiled_pdf.exists(), "DLG-Benchmark.pdf was not produced"

        dest_sub = SUBMISSION_DIR / "DLG-Benchmark.pdf"
        dest_manu = MANUSCRIPT_OUT_DIR / "DLG-Benchmark.pdf"
        shutil.copy2(compiled_pdf, dest_sub)
        shutil.copy2(compiled_pdf, dest_manu)

        # Check page count
        page_matches = re.findall(r"Output written on DLG-Benchmark\.pdf \((\d+) pages", log_content)
        page_count = int(page_matches[0]) if page_matches else 0
        log.info(f"Compilation SUCCESS! Pages: {page_count}, Size: {dest_sub.stat().st_size / 1024:.1f} KB")
        assert page_count >= 29, f"Expected at least 29 pages, got {page_count}"

    log.info("Submission bundle verification PASSED 100%!")


if __name__ == "__main__":
    create_submission_zip()
    test_compile_bundle()
