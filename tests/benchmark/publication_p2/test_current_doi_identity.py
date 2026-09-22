#!/usr/bin/env python3
"""
test_current_doi_identity.py

Round P2 Gate: Verifies strict DOI identity separation:
  - 10.20944/preprints202609.0848.v1 belongs exclusively to the preceding paper (DLG-GNN)
  - Current benchmark manuscript preprint_doi remains null pending Preprints.org posting
  - Preceding DOI cannot be assigned to current paper by update script
"""

from pathlib import Path
import json
import yaml
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_META = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m5" / "release" / "release_metadata.json"
CITATION_CFF = REPO_ROOT / "CITATION.cff"
PRECEDING_DOI = "10.20944/preprints202609.0848.v1"


def test_doi_identity_separation_p2():
    assert RELEASE_META.exists()
    data = json.loads(RELEASE_META.read_text(encoding="utf-8"))

    # Preceding paper DOI must be PRECEDING_DOI
    assert data.get("preceding_work_doi") == PRECEDING_DOI

    # Current benchmark paper DOI must be null pending Preprints deposit
    assert data.get("preprint_doi") is None or data.get("preprint_doi") == ""

    # CITATION.cff must not claim PRECEDING_DOI as top-level DOI
    assert CITATION_CFF.exists()
    cff_data = yaml.safe_load(CITATION_CFF.read_text(encoding="utf-8"))
    top_doi = cff_data.get("doi")
    assert top_doi != PRECEDING_DOI, "CITATION.cff falsely claims preceding DOI as its top-level DOI!"
