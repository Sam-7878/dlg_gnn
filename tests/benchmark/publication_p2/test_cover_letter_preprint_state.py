#!/usr/bin/env python3
"""
test_cover_letter_preprint_state.py

Round P2 Gate: Verifies that the MDPI cover letter declares the preprint
posting status accurately and that the post-deposit DOI update tool operates
with strict guard verification.
"""

from pathlib import Path
import subprocess
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COVER_LETTER = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "cover_letter.md"
UPDATE_SCRIPT = REPO_ROOT / "scripts" / "publication" / "update_preprint_doi.py"
PRECEDING_DOI = "10.20944/preprints202609.0848.v1"


def test_cover_letter_preprint_statement():
    assert COVER_LETTER.exists(), f"Missing {COVER_LETTER}"
    content = COVER_LETTER.read_text(encoding="utf-8")

    assert "Preprints.org" in content
    assert "CC BY 4.0" in content
    assert "Companion Repository" in content
    assert "v1.0.0-preprint" in content


def test_update_script_dry_run_and_guards():
    assert UPDATE_SCRIPT.exists(), f"Missing {UPDATE_SCRIPT}"

    # Dry-run test with dummy new DOI
    res = subprocess.run(
        [sys.executable, str(UPDATE_SCRIPT), "--doi", "10.20944/preprints202609.9999.v1", "--dry-run"],
        cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert res.returncode == 0, f"Dry-run failed: {res.stderr}"
    assert "[DRY-RUN] Validation successful." in res.stdout

    # Test guard rejecting preceding DOI
    res_bad = subprocess.run(
        [sys.executable, str(UPDATE_SCRIPT), "--doi", PRECEDING_DOI, "--dry-run"],
        cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert res_bad.returncode != 0, "Update script must refuse to set the preceding paper's DOI!"
