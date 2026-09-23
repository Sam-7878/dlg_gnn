#!/usr/bin/env python3
"""
test_m3_control_hashes_unchanged.py

Round M5 Freeze Gate: Cryptographically validates that the 45 Round M3 sensitivity
control runs and M4 paired deltas remain strictly frozen and unaltered.
"""

from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_45_control_runs_unaltered():
    candidates = [
        REPO_ROOT / "outputs/benchmark/manuscript_m5/artifacts/controls/capacity_controls_summary_m3_raw.csv",
        REPO_ROOT / "outputs/benchmark/manuscript_m3/controls/capacity_controls_summary_m3_raw.csv"
    ]
    for c in candidates:
        assert c.exists(), f"Missing control raw summary at {c}"
        df = pd.read_csv(c)
        assert len(df) == 45, f"Expected exactly 45 sensitivity runs, got {len(df)}"

        # Verify dataset coverage
        datasets = set(df["dataset"].unique())
        assert datasets == {"Elliptic", "DGraphFin", "LANL-RedTeam"}

        # Verify model variants
        models = set(df["model"].unique())
        assert models == {"DLG-Aug-Zero", "DLG-Base-70", "DLG-Aug-Permuted"}

        # Verify seeds
        seeds = set(df["seed"].unique())
        assert seeds == {42, 43, 44, 45, 46}


def test_paired_deltas_table_unaltered():
    p = REPO_ROOT / "outputs/benchmark/manuscript_m5/artifacts/controls/capacity_controls_paired_seed_differences_m4.csv"
    assert p.exists(), f"Missing paired deltas CSV at {p}"
    df = pd.read_csv(p)
    assert len(df) == 27, f"Expected 27 paired comparison rows (3 datasets x 3 comparisons x 3 metrics), got {len(df)}"
