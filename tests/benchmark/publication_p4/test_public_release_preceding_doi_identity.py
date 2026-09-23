#!/usr/bin/env python3
"""
test_public_release_preceding_doi_identity.py

Round P4 Publication Gate:
Verifies that:
1. Preceding work DOI is correctly identified as '10.20944/preprints202609.0848.v1'.
2. Current manuscript DOI is null prior to Preprints.org issuance.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PRECEDING_DOI = "10.20944/preprints202609.0848.v1"


def test_preceding_doi_in_release_metadata():
    meta_path = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/release_metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("preceding_work_doi") == PRECEDING_DOI
    assert meta.get("preprint_doi") is None


def test_preceding_doi_in_citation_cff():
    cff_path = REPO_ROOT / "CITATION.cff"
    assert cff_path.exists()
    content = cff_path.read_text(encoding="utf-8")
    assert PRECEDING_DOI in content


def test_preceding_doi_in_root_readme():
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")
    assert PRECEDING_DOI in content
