#!/usr/bin/env python3
"""
test_round5_hashes_unchanged.py

Round M5 Freeze Gate: Cryptographically validates that the primary Round 5 benchmark
raw results and support matrix remain 100% untouched and match their frozen hashes.
"""

import hashlib
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

FROZEN_RAW_HASH = "39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c"
FROZEN_SUPPORT_HASH = "c58dbca9a9e1ed14dfc025075820a3ad745f6cb70be77764c265d90af3522914"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_round5_primary_raw_hash():
    candidates = [
        REPO_ROOT / "outputs/benchmark/manuscript_m5/artifacts/primary/benchmark_raw.csv",
        REPO_ROOT / "outputs/benchmark/sci_round5_final/raw/benchmark_raw.csv"
    ]
    for c in candidates:
        assert c.exists(), f"Missing primary raw CSV at {c}"
        assert sha256_file(c) == FROZEN_RAW_HASH, f"Hash mismatch for {c}"

    df = pd.read_csv(candidates[0])
    assert len(df) == 355, f"Expected exactly 355 successful primary runs, got {len(df)}"


def test_round5_support_matrix_hash():
    candidates = [
        REPO_ROOT / "outputs/benchmark/manuscript_m5/artifacts/primary/model_dataset_support_matrix.csv",
        REPO_ROOT / "outputs/benchmark/sci_round5_final/manifests/model_dataset_support_matrix_v2.csv"
    ]
    for c in candidates:
        assert c.exists(), f"Missing support matrix at {c}"
        assert sha256_file(c) == FROZEN_SUPPORT_HASH, f"Hash mismatch for {c}"

    df = pd.read_csv(candidates[0])
    assert len(df) == 80, f"Expected 80 model-dataset pairs, got {len(df)}"
    supported_count = int(df["supported"].sum())
    assert supported_count == 71, f"Expected 71 supported pairs, got {supported_count}"
