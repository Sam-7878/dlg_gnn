#!/usr/bin/env python3
"""
test_graphical_abstract_support_statement_separated.py

Round P3 Gate: Verifies that the Graphical Abstract separates the baseline
support range (50% - 100%) from the overall suite pair count (71/80),
avoiding conflation of baseline-only ranges with the overall 80-pair count.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "manuscript" / "generate_graphical_abstract_p1.py"


def test_support_statement_separated():
    assert SCRIPT_PATH.exists(), f"Missing file: {SCRIPT_PATH}"
    content = SCRIPT_PATH.read_text(encoding="utf-8")

    # Ambiguous merged phrase must not be present
    assert "baseline support spans 50%-100% (71/80 supported pairs)" not in content
    assert "baseline support spans 50% - 100% (71/80 supported pairs)" not in content

    # Separated statements must be present
    assert "Baseline support spans 50%" in content
    assert "71/80 model-dataset pairs are supported" in content or "71/80 model-dataset pairs are exactly supported" in content
