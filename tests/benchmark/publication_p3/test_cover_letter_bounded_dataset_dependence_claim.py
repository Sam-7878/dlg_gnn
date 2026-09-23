#!/usr/bin/env python3
"""
test_cover_letter_bounded_dataset_dependence_claim.py

Round P3 Gate: Verifies that the cover letter softens overclaims regarding
dataset dependence (eliminating 'strictly' and 'demonstrates') and adopts
suitably bounded language: "the primary benchmark and targeted sensitivity
analyses show that the utility of local neighborhood augmentation is dataset dependent".
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COVER_LETTER = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "cover_letter.md"


def test_cover_letter_bounded_claim():
    assert COVER_LETTER.exists(), f"Missing file: {COVER_LETTER}"
    content = COVER_LETTER.read_text(encoding="utf-8")

    # Erroneous unhedged claim must be absent
    assert "strictly dataset-dependent" not in content
    assert "strictly dataset dependent" not in content

    # Bounded phrasing must be present
    assert "utility of local neighborhood augmentation is dataset dependent" in content or "utility of local augmentation is dataset dependent" in content
    assert "show that the utility" in content
