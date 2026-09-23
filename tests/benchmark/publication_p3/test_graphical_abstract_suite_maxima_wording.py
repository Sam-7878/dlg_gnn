#!/usr/bin/env python3
"""
test_graphical_abstract_suite_maxima_wording.py

Round P3 Gate: Verifies that the Graphical Abstract avoids phrasing that
implies a single graph attains both maxima simultaneously, using precise
suite-maxima wording ("Suite maxima: 1.23M nodes and 114.9M edges").
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "manuscript" / "generate_graphical_abstract_p1.py"


def test_suite_maxima_wording():
    assert SCRIPT_PATH.exists(), f"Missing file: {SCRIPT_PATH}"
    content = SCRIPT_PATH.read_text(encoding="utf-8")

    # Ambiguous phrasing must be absent
    assert "up to 1.2M nodes, 114M edges" not in content
    assert "up to 1.23M nodes, 114.9M edges" not in content

    # Precise wording must be present
    assert "Suite maxima: 1.23M nodes and 114.9M edges" in content or "Max N = 1.23M nodes; Max E = 114.9M" in content
