#!/usr/bin/env python3
"""
test_graphical_abstract_support_rate_wording.py

Round P2 Gate: Verifies that the Graphical Abstract does not falsely claim
that all baselines have 50%-90% support (since DOMINANT, CoLA, and OCGNN have 100% support),
and correctly frames baseline support as spanning 50%-100%.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "manuscript" / "generate_graphical_abstract_p1.py"


def test_support_rate_wording_accurate():
    assert SCRIPT_PATH.exists()
    content = SCRIPT_PATH.read_text(encoding="utf-8")

    # Forbidden inaccurate claim
    assert "while baselines exhibit 50% - 90% operational support" not in content
    assert "while baselines exhibit 50%-90% operational support" not in content

    # Required accurate claim
    assert "50% - 100%" in content or "50%-100%" in content or "support all 10 datasets" in content
    assert "DLG-Base & DLG-Aug support 10/10" in content or "support all 10 primary datasets" in content
