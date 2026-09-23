#!/usr/bin/env python3
"""
test_graphical_abstract_rank_scope.py

Round P3 Gate: Verifies that the Graphical Abstract makes the scope of the
ranking result explicit (bounded to the 7-dataset fraud-oriented common subset
across 5 fully supported models) rather than claiming a broad or universal rank 1.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "manuscript" / "generate_graphical_abstract_p1.py"


def test_rank_scope_explicit():
    assert SCRIPT_PATH.exists(), f"Missing file: {SCRIPT_PATH}"
    content = SCRIPT_PATH.read_text(encoding="utf-8")

    # Scope must be explicitly declared
    assert "Best Average Rank in Fraud-Oriented Common Subset" in content
    assert "across 7 fraud graphs (5 models)" in content
    assert "1.71" in content
    assert "1.86" in content
