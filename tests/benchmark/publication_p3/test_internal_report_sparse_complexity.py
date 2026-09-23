#!/usr/bin/env python3
"""
test_internal_report_sparse_complexity.py

Round P3 Gate: Verifies that internal publication audit report 07
corrects the legacy 'O(|E|) complexity' overclaim and accurately describes
exact sparse reconstruction: avoiding O(N^2) dense structural storage,
with O(|E|d + Nd^2) arithmetic for the Gram-based linear decoder.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_07 = REPO_ROOT / "publication" / "benchmark" / "reports" / "07_mdpi_special_issue_submission_readiness.md"


def test_internal_report_07_complexity_reconciled():
    assert REPORT_07.exists(), f"Missing file: {REPORT_07}"
    content = REPORT_07.read_text(encoding="utf-8")

    # Erroneous unhedged complexity must be absent
    assert "O(|E|) complexity" not in content
    assert r"$\mathcal{O}(|\mathcal{E}|)$ complexity" not in content

    # Accurate storage avoiding O(N^2) and arithmetic complexity must be present
    assert "dense structural storage" in content
    assert "O(N^2)" in content or r"\mathcal{O}(N^2)" in content
    assert "Nd^2" in content
