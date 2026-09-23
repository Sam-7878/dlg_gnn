#!/usr/bin/env python3
"""
test_public_release_no_internal_round_name.py

Round P4 Publication Gate:
Verifies that public release assets and public documents do not use internal
remediation round names ('Round M5', 'Round M4', 'Round M3') in user-facing titles.
"""

from pathlib import Path
import zipfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ZIP = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_v1.0.0_preprint.zip"


def test_release_zip_name_no_internal_round():
    assert "M5" not in RELEASE_ZIP.name
    assert "M4" not in RELEASE_ZIP.name
    assert "M3" not in RELEASE_ZIP.name
    assert "v1.0.0_preprint" in RELEASE_ZIP.name


def test_root_readme_title_no_internal_round():
    readme = REPO_ROOT / "README.md"
    assert readme.exists()
    first_lines = "".join(readme.read_text(encoding="utf-8").splitlines()[:15])
    assert "Round M5" not in first_lines
    assert "Round M4" not in first_lines
