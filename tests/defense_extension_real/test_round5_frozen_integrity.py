#!/usr/bin/env python3
"""
Test: Verification that Round 5 frozen benchmark results (10 datasets, 355 runs)
remain completely unchanged and intact during D3 execution.
"""
import hashlib
from pathlib import Path
import pytest

FROZEN_RAW_SHA256 = "39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c"
FROZEN_SUPPORT_SHA256 = "c58dbca9a9e1ed14dfc025075820a3ad745f6cb70be77764c265d90af3522914"

ROUND5_RAW_PATH = Path("outputs/sci_round5_final/raw/benchmark_raw.csv")
ROUND5_SUPPORT_PATH = Path("outputs/sci_round5_final/manifests/model_dataset_support_matrix_v2.csv")

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

def test_round5_raw_results_integrity():
    """Verify that Round 5 raw_results.json matches the exact frozen SHA-256."""
    if not ROUND5_RAW_PATH.exists():
        pytest.skip(f"Round 5 raw results file not found at {ROUND5_RAW_PATH}")
    actual_sha = compute_sha256(ROUND5_RAW_PATH)
    assert actual_sha == FROZEN_RAW_SHA256, (
        f"Round 5 raw_results.json hash mismatch! Expected {FROZEN_RAW_SHA256}, got {actual_sha}"
    )

def test_round5_support_matrix_integrity():
    """Verify that Round 5 support_matrix.json matches the exact frozen SHA-256."""
    if not ROUND5_SUPPORT_PATH.exists():
        pytest.skip(f"Round 5 support matrix file not found at {ROUND5_SUPPORT_PATH}")
    actual_sha = compute_sha256(ROUND5_SUPPORT_PATH)
    assert actual_sha == FROZEN_SUPPORT_SHA256, (
        f"Round 5 support_matrix.json hash mismatch! Expected {FROZEN_SUPPORT_SHA256}, got {actual_sha}"
    )
