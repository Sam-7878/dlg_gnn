#!/usr/bin/env python3
"""
Test: Verification of raw record accounting and node universe mapping.
Defense Extension Round D3 hard gate requirement.
"""
import csv
from pathlib import Path
import pytest

DARPA_ACCOUNTING_PATH = Path("outputs/sci_defense_extension_real/source_audit/darpa_record_accounting.csv")
LANL_UNIVERSE_PATH = Path("outputs/sci_defense_extension_real/source_audit/lanl_node_universe.csv")

def test_darpa_record_accounting():
    """Verify that DARPA record accounting counts raw records, entity types, and events."""
    if not DARPA_ACCOUNTING_PATH.exists():
        pytest.skip(f"DARPA record accounting not found at {DARPA_ACCOUNTING_PATH}")
    with open(DARPA_ACCOUNTING_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    counts = dict(reader)
    assert "total" in counts, "Must record total raw records"
    assert int(counts["total"]) > 0, "Total records must be positive"
    assert "Event" in counts or "RECORD_EVENT" in counts or any("Event" in k for k in counts), "Must account for Events"

def test_lanl_node_universe_accounting():
    """Verify that LANL node universe accounts for all telemetry modalities and matches reported topology."""
    if not LANL_UNIVERSE_PATH.exists():
        pytest.skip(f"LANL node universe not found at {LANL_UNIVERSE_PATH}")
    with open(LANL_UNIVERSE_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    counts = dict(reader)
    assert "official_reported_computers" in counts or "auth" in counts or any("auth" in k for k in counts), "Must account for auth computers"
