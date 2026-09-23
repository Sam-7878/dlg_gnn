#!/usr/bin/env python3
"""
test_repository_public_at_preprint_release.py

Round P1 Freeze Gate: Verifies that the public release bundle, metadata,
and repository release tag are properly structured for public release upon
Preprints.org deposit.
"""

from pathlib import Path
import json
import zipfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m5" / "release"
RELEASE_ZIP = RELEASE_DIR / "DLG_GNN_Benchmark_M5_Release.zip"
RELEASE_META = RELEASE_DIR / "release_metadata.json"


def test_release_bundle_exists_and_valid():
    assert RELEASE_ZIP.exists(), f"Missing release archive: {RELEASE_ZIP}"
    assert RELEASE_ZIP.stat().st_size > 10 * 1024, "Release archive suspiciously small"

    with zipfile.ZipFile(RELEASE_ZIP, "r") as zf:
        namelist = zf.namelist()
        assert any("release_metadata.json" in n for n in namelist)
        assert any("environment_manifest.json" in n for n in namelist)
        assert any("run_sci_round5_final.py" in n for n in namelist)
        assert any("reproduce_frozen_artifacts.py" in n for n in namelist)


def test_release_metadata_public_tag_and_url():
    assert RELEASE_META.exists(), f"Missing release metadata: {RELEASE_META}"
    data = json.loads(RELEASE_META.read_text(encoding="utf-8"))

    assert data.get("version") == "1.0.0-p1"
    assert data.get("git_tag") == "v1.0.0-preprint"

    repo_url = data.get("repository_url", "")
    assert "github.com" in repo_url, f"Invalid repository URL: {repo_url}"
    assert data.get("is_public_release") is True
