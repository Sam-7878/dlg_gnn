#!/usr/bin/env python3
"""
test_current_preprint_doi_not_preceding_paper.py

Round P1 Freeze Gate: Verifies that the preceding paper's DOI
(10.20944/preprints202609.0848.v1) is exclusively attributed to the foundational
preceding work (Park & Kim, 2026, DLG-GNN) and never claimed or conflated as the DOI
of the current benchmark paper (which remains null / preprint-forthcoming).
"""

import json
from pathlib import Path
import yaml
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PRECEDING_DOI = "10.20944/preprints202609.0848.v1"
RELEASE_META = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m5" / "release" / "release_metadata.json"
CITATION_CFF = REPO_ROOT / "CITATION.cff"
PREPRINT_TEX = REPO_ROOT / "publication" / "benchmark" / "preprints" / "DLG-Benchmark-Preprint.tex"
MDPI_TEX = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "DLG-Benchmark.tex"
BIB_FILE = REPO_ROOT / "publication" / "benchmark" / "preprints" / "references.bib"


def test_release_metadata_doi_separation():
    assert RELEASE_META.exists(), f"Missing {RELEASE_META}"
    data = json.loads(RELEASE_META.read_text(encoding="utf-8"))

    assert data.get("preceding_work_doi") == PRECEDING_DOI
    assert data.get("preprint_doi") is None, "Benchmark preprint_doi must be null pending deposit"
    assert data.get("preprint_status") == "preprint-forthcoming"


def test_citation_cff_doi_integrity():
    assert CITATION_CFF.exists(), f"Missing {CITATION_CFF}"
    cff = yaml.safe_load(CITATION_CFF.read_text(encoding="utf-8"))

    # Top level DOI must not be the preceding DOI
    assert cff.get("doi") != PRECEDING_DOI

    # References must include preceding work with PRECEDING_DOI
    refs = cff.get("references", [])
    matched = [r for r in refs if r.get("doi") == PRECEDING_DOI]
    assert len(matched) == 1, f"Preceding reference with DOI {PRECEDING_DOI} must exist in CITATION.cff"
    assert "DLG-GNN" in matched[0].get("title", "")


def test_manuscript_files_do_not_claim_preceding_doi():
    for tex_path in [PREPRINT_TEX, MDPI_TEX]:
        assert tex_path.exists()
        content = tex_path.read_text(encoding="utf-8")
        # Ensure PRECEDING_DOI is not directly in the body as this paper's DOI
        assert PRECEDING_DOI not in content, (
            f"{tex_path.name} incorrectly contains preceding DOI {PRECEDING_DOI} in source body"
        )

    # In references.bib, it should appear exclusively under park2026dlg
    assert BIB_FILE.exists()
    bib_text = BIB_FILE.read_text(encoding="utf-8")
    assert PRECEDING_DOI in bib_text
    assert "park2026dlg" in bib_text
