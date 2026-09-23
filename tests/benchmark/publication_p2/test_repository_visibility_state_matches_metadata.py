#!/usr/bin/env python3
"""
test_repository_visibility_state_matches_metadata.py

Round P2 Gate: Verifies that release metadata, public tags, and repository
visibility audit documentation are mutually consistent and prepared for
immediate logged-out access upon preprint deposit.
"""

from pathlib import Path
import json
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_META = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m5" / "release" / "release_metadata.json"
VISIBILITY_DOC = REPO_ROOT / "publication" / "benchmark" / "reports" / "08_repository_external_visibility_check.md"
RELEASE_ZIP = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m5" / "release" / "DLG_GNN_Benchmark_M5_Release.zip"


def test_visibility_metadata_consistency():
    assert RELEASE_META.exists(), f"Missing {RELEASE_META}"
    assert VISIBILITY_DOC.exists(), f"Missing {VISIBILITY_DOC}"
    assert RELEASE_ZIP.exists(), f"Missing {RELEASE_ZIP}"

    meta = json.loads(RELEASE_META.read_text(encoding="utf-8"))
    doc_text = VISIBILITY_DOC.read_text(encoding="utf-8")

    git_tag = meta.get("git_tag")
    assert git_tag == "v1.0.0-preprint"
    assert git_tag in doc_text

    repo_url = meta.get("repository_url")
    assert repo_url in doc_text

    assert RELEASE_ZIP.stat().st_size > 10 * 1024
