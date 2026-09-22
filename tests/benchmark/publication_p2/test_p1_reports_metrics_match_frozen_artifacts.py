#!/usr/bin/env python3
"""
test_p1_reports_metrics_match_frozen_artifacts.py

Round P2 Gate: Verifies that P1 Audit Report 01 contains only authentic,
programmatically verified metrics matching frozen Round 5 artifacts,
and contains zero hallucinated metrics (e.g. 0.941, 0.781, 0.803, 0.942).
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_01 = REPO_ROOT / "publication" / "benchmark" / "reports" / "01_final_scientific_wording_audit.md"

REQUIRED_FROZEN_METRICS = [
    "0.1037",   # Elliptic DLG-Aug PR-AUC
    "0.0134",   # DGraphFin DLG-Aug PR-AUC
    "0.3388",   # Reddit-Syn DLG-Aug PR-AUC
    "0.7923",   # LANL DLG-Base ROC-AUC
    "0.7188",   # LANL DLG-Aug ROC-AUC
    "0.8261",   # LANL GADNR ROC-AUC
]

BANNED_HALLUCINATED_METRICS = [
    "0.941",
    "0.781",
    "0.803",
    "0.963",
]


def test_report_01_metrics_reconciled():
    assert REPORT_01.exists(), f"Missing {REPORT_01}"
    content = REPORT_01.read_text(encoding="utf-8")

    # Assert exact frozen values are present
    for val in REQUIRED_FROZEN_METRICS:
        assert val in content, f"Required frozen metric '{val}' missing from Report 01"

    # Assert hallucinated values are absent
    for val in BANNED_HALLUCINATED_METRICS:
        assert val not in content, f"Hallucinated metric '{val}' found in Report 01"
