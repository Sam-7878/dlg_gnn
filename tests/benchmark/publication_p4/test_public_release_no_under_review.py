#!/usr/bin/env python3
"""
test_public_release_no_under_review.py

Round P4 Publication Gate:
Verifies that no public release metadata, README, INSTALL, CITATION,
release_metadata.json, or artifacts in the release ZIP claim 'Under Review'.
"""

from pathlib import Path
import zipfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ZIP = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_v1.0.0_preprint.zip"


def test_root_files_have_no_under_review():
    target_files = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "INSTALL.md",
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "outputs/benchmark/manuscript_m5/release/release_metadata.json",
    ]
    for tf in target_files:
        if tf.exists():
            text = tf.read_text(encoding="utf-8").lower()
            assert "under review" not in text, f"Found 'under review' in {tf.name}"


def test_release_zip_has_no_under_review():
    assert RELEASE_ZIP.exists(), f"Missing {RELEASE_ZIP}"
    with zipfile.ZipFile(RELEASE_ZIP, "r") as zf:
        for info in zf.infolist():
            if info.filename.endswith((".md", ".json", ".cff", ".tex", ".py", ".toml", ".yml", ".txt")):
                content = zf.read(info.filename).decode("utf-8", errors="ignore").lower()
                assert "under review" not in content, f"Found 'under review' inside release zip: {info.filename}"
