#!/usr/bin/env python3
"""
test_no_darpa_theia_fingate_in_publication_docs.py

Round P2 Gate: Verifies that no stale references to DARPA, THEIA, Theia,
FinGate, FinGate-28k, or '14 baseline' / '14-baseline' appear in publication reports,
cover letters, or submission documentation.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLICATIONS_DIR = REPO_ROOT / "publication"

FORBIDDEN_TERMS = [
    "darpa",
    "theia",
    "fingate",
    "fingate-28k",
    "14 baseline",
    "14-baseline",
    "14 baselines",
]


def test_no_stale_terms_in_publication_reports_and_cover_letter():
    files_to_check = []
    # All reports in publication/benchmark/reports/
    reports_dir = PUBLICATIONS_DIR / "benchmark" / "reports"
    if reports_dir.exists():
        files_to_check.extend([p for p in reports_dir.glob("*.md")])

    # Cover letter
    cover_letter = PUBLICATIONS_DIR / "benchmark" / "mdpi" / "cover_letter.md"
    if cover_letter.exists():
        files_to_check.append(cover_letter)

    # Graphical abstract script
    ga_script = REPO_ROOT / "scripts" / "manuscript" / "generate_graphical_abstract_p1.py"
    if ga_script.exists():
        files_to_check.append(ga_script)

    assert len(files_to_check) > 0, "No files found to inspect"

    for file_path in files_to_check:
        content = file_path.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_TERMS:
            assert term not in content, f"Forbidden stale term '{term}' found in {file_path.name}"
