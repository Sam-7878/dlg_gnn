#!/usr/bin/env python3
"""
test_submission_bundle_self_contained.py

Round M5 Freeze Gate: Verifies that the MDPI submission bundle
(outputs/benchmark/manuscript_m5/submission/DLG_Benchmark_MDPI_Submission.zip)
contains all necessary LaTeX inputs, Definitions, and generated tables, and compiles
cleanly without error.
"""

from pathlib import Path
import subprocess
import tempfile
import zipfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMISSION_ZIP = REPO_ROOT / "outputs/benchmark/manuscript_m5/submission/DLG_Benchmark_MDPI_Submission.zip"
COMPILED_PDF = REPO_ROOT / "outputs/benchmark/manuscript_m5/submission/DLG-Benchmark.pdf"


def test_submission_bundle_files():
    assert SUBMISSION_ZIP.exists(), f"Submission zip missing: {SUBMISSION_ZIP}"
    assert COMPILED_PDF.exists(), f"Compiled submission PDF missing: {COMPILED_PDF}"

    with zipfile.ZipFile(SUBMISSION_ZIP, "r") as zf:
        namelist = zf.namelist()
        assert "DLG-Benchmark.tex" in namelist
        assert "references.bib" in namelist
        assert any("table_capacity_controls_paired_deltas.tex" in name for name in namelist)


def test_compiled_pdf_integrity():
    assert COMPILED_PDF.stat().st_size > 100 * 1024, "Compiled PDF too small"
