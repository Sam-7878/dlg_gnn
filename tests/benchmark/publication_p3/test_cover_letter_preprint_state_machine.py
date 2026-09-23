#!/usr/bin/env python3
"""
test_cover_letter_preprint_state_machine.py

Round P3 Gate: Verifies that the MDPI cover letter manages preprint state
cleanly without conflating pending deposit with completed posting:
  - Declares preprints deposit initiated / forthcoming
  - Declares official companion repository URL: https://github.com/Sam-7878/dlg_gnn
  - Avoids asserting non-existent completed DOI prior to deposit confirmation
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COVER_LETTER = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "cover_letter.md"


def test_cover_letter_preprint_state_and_url():
    assert COVER_LETTER.exists(), f"Missing file: {COVER_LETTER}"
    content = COVER_LETTER.read_text(encoding="utf-8")

    # State declaration
    assert "deposit initiated" in content or "deposit pending" in content or "preprint-forthcoming" in content

    # Authoritative repository URL
    assert "https://github.com/Sam-7878/dlg_gnn" in content
    assert "v1.0.0-preprint" in content

    # Should not claim completed DOI
    assert "10.20944/preprints202609.0848" not in content  # Preceding paper DOI must not be reused for this paper
