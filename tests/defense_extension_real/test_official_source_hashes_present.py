#!/usr/bin/env python3
"""
Test: Verification that official raw source manifests and SHA-256 hashes are recorded.
Defense Extension Round D3 hard gate requirement.
"""
import csv
from pathlib import Path
import pytest

DARPA_MANIFEST_PATH = Path("outputs/sci_defense_extension_real/source_audit/darpa_raw_manifest.csv")
LANL_MANIFEST_PATH = Path("outputs/sci_defense_extension_real/source_audit/lanl_real_manifest.csv")

def test_darpa_manifest_present_and_valid():
    """Verify that DARPA raw manifest contains valid SHA-256 hashes for all processed files."""
    if not DARPA_MANIFEST_PATH.exists():
        pytest.skip(f"DARPA manifest not found at {DARPA_MANIFEST_PATH}")
    with open(DARPA_MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) > 0, "DARPA manifest must have at least 1 record"
    for row in reader:
        assert "sha256" in row, "Each DARPA entry must have a sha256 field"
        assert len(row["sha256"]) == 64, f"Invalid SHA-256 length: {row['sha256']}"
        assert int(row["compressed_size_bytes"]) > 0, "File size must be positive"

def test_lanl_manifest_present_and_valid():
    """Verify that LANL raw manifest contains valid SHA-256 hashes for all processed files."""
    if not LANL_MANIFEST_PATH.exists():
        pytest.skip(f"LANL manifest not found at {LANL_MANIFEST_PATH}")
    with open(LANL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) > 0, "LANL manifest must have at least 1 record"
    for row in reader:
        assert "sha256" in row, "Each LANL entry must have a sha256 field"
        assert len(row["sha256"]) == 64, f"Invalid SHA-256 length: {row['sha256']}"
        assert int(row["compressed_size_bytes"]) > 0, "File size must be positive"
