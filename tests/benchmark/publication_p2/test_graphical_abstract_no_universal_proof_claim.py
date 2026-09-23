#!/usr/bin/env python3
"""
test_graphical_abstract_no_universal_proof_claim.py

Round P2 Gate: Verifies that the Graphical Abstract eliminates overclaims
such as 'Proves' and 'must', framing local augmentation as conditional rather
than universal.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "manuscript" / "generate_graphical_abstract_p1.py"


def test_no_proves_claim_in_graphical_abstract():
    assert SCRIPT_PATH.exists()
    content = SCRIPT_PATH.read_text(encoding="utf-8")

    # Forbidden overclaim words in conclusion box
    assert "Proves neighbourhood augmentation" not in content
    assert "proves neighborhood augmentation" not in content
    assert "must be adaptively" not in content

    # Required balanced phrasing
    assert "Local Augmentation is Conditional, Not Universal" in content
    assert "motivate adaptive, graph-dependent use" in content
