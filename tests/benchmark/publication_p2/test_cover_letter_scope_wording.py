#!/usr/bin/env python3
"""
test_cover_letter_scope_wording.py

Round P2 Gate: Verifies that the MDPI cover letter exactly mirrors the official
Special Issue topics, describes anomaly detection as the application domain,
and uses verified exact-sparse reformulation terminology.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COVER_LETTER = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "cover_letter.md"


def test_special_issue_scope_bullets_accuracy():
    assert COVER_LETTER.exists(), f"Missing {COVER_LETTER}"
    content = COVER_LETTER.read_text(encoding="utf-8")

    # Required official Special Issue scope items
    assert "Scalable and efficient GNN architectures" in content
    assert "Self-supervised and unsupervised graph learning" in content
    assert "Graph representation learning" in content
    assert "Financial modeling and other applied GNN settings" in content

    # Should not treat anomaly detection as an explicit SI bullet point
    assert "- *Financial modeling and anomaly detection*" not in content

    # Exact sparse reformulation wording
    assert "derive, implement, and numerically verify a mathematically equivalent sparse reformulation" in content
    assert "we develop and mathematically prove an exact sparse formulation" not in content
