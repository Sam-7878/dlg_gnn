#!/usr/bin/env python3
"""
test_no_mdpi_branding_in_preprint.py

Round P1 Freeze Gate: Verifies that the Preprints.org submission manuscript
(DLG-Benchmark-Preprint.tex) and bundle (DLG_Benchmark_Preprints_Submission.zip)
contain zero occurrences of MDPI logos, publisher headers, or journal-specific branding.
"""

from pathlib import Path
import zipfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PREPRINTS_DIR = REPO_ROOT / "publication" / "benchmark" / "preprints"
PREPRINT_TEX = PREPRINTS_DIR / "DLG-Benchmark-Preprint.tex"
PREPRINT_ZIP = PREPRINTS_DIR / "DLG_Benchmark_Preprints_Submission.zip"
PREPRINT_PDF = PREPRINTS_DIR / "DLG-Benchmark-Preprint.pdf"


def test_preprint_tex_no_mdpi_branding():
    assert PREPRINT_TEX.exists()
    content = PREPRINT_TEX.read_text(encoding="utf-8").lower()

    banned_terms = [
        "mdpi.cls",
        "definitions/mdpi",
        "logo-mdpi",
        "applsci",
        "applied sciences",
        r"\pubvolume",
        r"\articlenumber",
        r"\issuenum",
        r"\datereceived",
        "academic editor",
    ]
    for term in banned_terms:
        assert term not in content, f"Banned publisher branding term '{term}' found in preprint LaTeX source!"


def test_preprint_zip_contains_no_mdpi_assets():
    assert PREPRINT_ZIP.exists()
    with zipfile.ZipFile(PREPRINT_ZIP, "r") as zf:
        namelist = [n.lower() for n in zf.namelist()]
        for name in namelist:
            assert "logo-mdpi" not in name, f"MDPI logo file found in preprint bundle: {name}"
            assert "mdpi.cls" not in name, f"MDPI class file found in preprint bundle: {name}"
            assert "logo-ccby" not in name, f"MDPI logo file found in preprint bundle: {name}"


def test_preprint_pdf_exists_and_valid():
    assert PREPRINT_PDF.exists()
    assert PREPRINT_PDF.stat().st_size > 100000, "Preprint PDF size is suspiciously small"
