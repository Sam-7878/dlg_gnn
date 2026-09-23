#!/usr/bin/env python3
"""
test_no_false_no_scientific_interpretation_claim.py

Round P3 Gate: Verifies that the attestation matrix eliminates false
categorical denials such as "No AI model generated scientific interpretations"
and instead accurately describes AI review suggestions while affirming independent
author verification, calculation ownership, and final interpretive decisions.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ATTESTATION_MD = REPO_ROOT / "publication" / "benchmark" / "ai_use_author_attestation.md"


def test_no_categorical_denial_of_scientific_interpretation():
    assert ATTESTATION_MD.exists(), f"Missing file: {ATTESTATION_MD}"
    content = ATTESTATION_MD.read_text(encoding="utf-8")

    # Erroneous categorical denial must be absent
    assert "No AI model generated scientific interpretations" not in content
    assert "no AI model generated scientific interpretations" not in content.lower()

    # Nuanced boundary must be present
    assert "AI-assisted tools were used to provide drafting" in content
    assert "scientific-review" in content
    assert "No AI tool acted as an autonomous scientific authority" in content
    assert "All numerical analyses, statistical outputs, source verifications, interpretation choices, and final scientific conclusions were independently checked and approved by the authors" in content
