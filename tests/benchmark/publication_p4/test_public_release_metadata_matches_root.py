#!/usr/bin/env python3
"""
test_public_release_metadata_matches_root.py

Round P4 Publication Gate:
Verifies that metadata declared in release_metadata.json matches CITATION.cff
and the repository root configuration.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_metadata_parity():
    meta_path = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/release_metadata.json"
    cff_path = REPO_ROOT / "CITATION.cff"
    assert meta_path.exists() and cff_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cff_text = cff_path.read_text(encoding="utf-8")

    assert meta["repository_url"] in cff_text
    assert meta["version"] in cff_text
    assert meta["preceding_work_doi"] in cff_text
    for author in meta["authors"]:
        assert author.split()[-1] in cff_text  # family name check
