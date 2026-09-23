#!/usr/bin/env python3
"""
test_repository_state_boolean_semantics.py

Round P3 Gate: Verifies that release metadata enforces clean boolean and enum
semantics without hybrid/staged strings, that release_state is 'public',
and that repository URLs and CITATION.cff are authoritative.
"""

from pathlib import Path
import json
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_META = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m5" / "release" / "release_metadata.json"
CITATION_CFF = REPO_ROOT / "CITATION.cff"


def test_release_metadata_semantics():
    assert RELEASE_META.exists(), f"Missing file: {RELEASE_META}"
    data = json.loads(RELEASE_META.read_text(encoding="utf-8"))

    # Boolean semantics
    assert isinstance(data.get("is_public_release"), bool)
    assert data.get("is_public_release") is True

    # State enum
    assert data.get("release_state") == "public"

    # URL
    assert data.get("repository_url") == "https://github.com/Sam-7878/dlg_gnn"


def test_citation_cff_authoritative():
    assert CITATION_CFF.exists(), f"Missing file: {CITATION_CFF}"
    content = CITATION_CFF.read_text(encoding="utf-8")

    assert "0009-0008-4056-3875" in content
    assert "0000-0002-2321-4475" in content
    assert "https://github.com/Sam-7878/dlg_gnn" in content
