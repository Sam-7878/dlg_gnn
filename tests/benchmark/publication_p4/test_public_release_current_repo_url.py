#!/usr/bin/env python3
"""
test_public_release_current_repo_url.py

Round P4 Publication Gate:
Verifies that all public-facing files and the release ZIP use the canonical
repository URL 'https://github.com/Sam-7878/dlg_gnn' and contain zero
stale 'goat-bank/dlg_gnn' URLs.
"""

from pathlib import Path
import zipfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ZIP = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_v1.0.0_preprint.zip"
CANONICAL_URL = "https://github.com/Sam-7878/dlg_gnn"
BANNED_URL = "https://github.com/goat-bank/dlg_gnn"


def test_root_files_use_canonical_url():
    target_files = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "INSTALL.md",
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "outputs/benchmark/manuscript_m5/release/release_metadata.json",
    ]
    for tf in target_files:
        assert tf.exists(), f"Missing {tf}"
        text = tf.read_text(encoding="utf-8")
        assert CANONICAL_URL in text, f"Expected {CANONICAL_URL} in {tf.name}"
        assert BANNED_URL not in text, f"Found stale {BANNED_URL} in {tf.name}"


def test_release_zip_uses_canonical_url():
    assert RELEASE_ZIP.exists(), f"Missing {RELEASE_ZIP}"
    with zipfile.ZipFile(RELEASE_ZIP, "r") as zf:
        for info in zf.infolist():
            if info.filename.endswith((".md", ".json", ".cff", ".tex", ".py", ".toml", ".yml", ".txt")):
                content = zf.read(info.filename).decode("utf-8", errors="ignore")
                assert BANNED_URL not in content, f"Found stale {BANNED_URL} in zip: {info.filename}"
