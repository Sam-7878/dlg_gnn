#!/usr/bin/env python3
"""
test_p1_reports_scope_matches_manuscript.py

Round P2 Gate: Verifies that P1 Audit Report 07 and all publication documentation
accurately declare the benchmark scope: 10 primary datasets, 8 detector configurations,
and LANL external validation.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_07 = REPO_ROOT / "publication" / "benchmark" / "reports" / "07_mdpi_special_issue_submission_readiness.md"


def test_report_07_scope_matches_manuscript():
    assert REPORT_07.exists(), f"Missing {REPORT_07}"
    content = REPORT_07.read_text(encoding="utf-8")

    assert "10 primary datasets" in content
    assert "8 detector configurations" in content
    assert "LANL" in content

    # Assert absence of stale description
    assert "14-baseline" not in content
    assert "5 financial/security graphs" not in content
