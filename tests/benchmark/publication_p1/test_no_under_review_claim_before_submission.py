#!/usr/bin/env python3
"""
test_no_under_review_claim_before_submission.py

Round P1 Freeze Gate: Verifies that public citation snippets, CITATION.cff,
README.md, and release metadata do not prematurely claim "Under Review" before
formal journal submission occurs.
"""

from pathlib import Path
import json
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CITATION_CFF = REPO_ROOT / "CITATION.cff"
README_MD = REPO_ROOT / "README.md"
RELEASE_META = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m5" / "release" / "release_metadata.json"
PREPRINT_TEX = REPO_ROOT / "publication" / "preprints" / "DLG-Benchmark-Preprint.tex"


def test_citation_cff_no_under_review():
    assert CITATION_CFF.exists()
    content = CITATION_CFF.read_text(encoding="utf-8")
    assert "Under Review" not in content, "CITATION.cff claims 'Under Review' prematurely!"


def test_readme_no_under_review():
    if README_MD.exists():
        content = README_MD.read_text(encoding="utf-8")
        assert "Under Review" not in content, "README.md claims 'Under Review' prematurely!"


def test_release_metadata_status():
    assert RELEASE_META.exists()
    data = json.loads(RELEASE_META.read_text(encoding="utf-8"))
    status = data.get("preprint_status", "")
    assert "under review" not in status.lower(), f"release_metadata.json has preprint_status='{status}'"
    assert status == "preprint-forthcoming"


def test_preprint_tex_no_under_review():
    assert PREPRINT_TEX.exists()
    content = PREPRINT_TEX.read_text(encoding="utf-8")
    assert "Under Review" not in content
    assert "under review" not in content
