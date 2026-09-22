#!/usr/bin/env python3
"""
test_preceding_paper_doi_identity.py

Round M5 Unit Test: Verifies that the preceding paper's DOI
(10.20944/preprints202609.0848.v1, Park & Kim, Sept 2026) is strictly attributed
to the preceding work (park2026dlg / preceding_work_doi) and is NEVER attributed
as the DOI of the current benchmark manuscript (which is null pending deposit).
"""

import json
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PRECEDING_DOI = "10.20944/preprints202609.0848.v1"


def test_release_metadata_doi_separation():
    meta_path = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/release_metadata.json"
    assert meta_path.exists()
    data = json.loads(meta_path.read_text(encoding="utf-8"))

    assert data["preceding_work_doi"] == PRECEDING_DOI
    assert data["preprint_doi"] is None, "Current paper preprint_doi must be null until deposited!"


def test_citation_cff_doi_separation():
    cff_path = REPO_ROOT / "CITATION.cff"
    assert cff_path.exists()
    data = yaml.safe_load(cff_path.read_text(encoding="utf-8"))

    # Top-level doi should NOT be the preceding DOI
    assert "doi" not in data or data["doi"] is None

    # Preceding work reference MUST have the preceding DOI
    refs = data.get("references", [])
    preceding_found = False
    for ref in refs:
        if ref.get("doi") == PRECEDING_DOI:
            preceding_found = True
            assert "DLG-GNN: Decoupled Local-to-Global Graph Neural Network for Scalable Blockchain Fraud Detection" in ref.get("title", "")
    assert preceding_found, f"Preceding reference with DOI {PRECEDING_DOI} must be declared in CITATION.cff"


def test_manuscript_bib_separation():
    bib_path = REPO_ROOT / "docs/papers/_42_Benchmark/references.bib"
    assert bib_path.exists()
    bib_content = bib_path.read_text(encoding="utf-8")

    # PRECEDING_DOI must be present under park2026dlg
    assert PRECEDING_DOI in bib_content
    assert "@article{park2026dlg" in bib_content or "@misc{park2026dlg" in bib_content
