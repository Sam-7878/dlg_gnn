"""Test defense support accounting and seed completeness."""
from pathlib import Path
import pandas as pd
import pytest

from gog_fraud.extensions.defense.defense_registry import DEFENSE_DATASETS

SUPPORTED_MODELS = ["DOMINANT", "AnomalyDAE", "CoLA", "CONAD", "GADNR", "OCGNN", "DLG-Base", "DLG-Aug"]
UNSUPPORTED_MODELS = []
EXPECTED_SEEDS = {42, 43, 44, 45, 46}


def test_defense_cell_runs_completed():
    raw_csv = Path("outputs/sci_defense_extension/raw/benchmark_raw.csv")
    if not raw_csv.exists():
        pytest.skip("benchmark_raw.csv not generated yet")

    df = pd.read_csv(raw_csv)
    assert not df.empty, "benchmark_raw.csv is empty"
    assert len(df) == 80, f"Expected 80 completed runs (8 models x 2 datasets x 5 seeds), got {len(df)}"
    assert "benchmark_origin" in df.columns
    assert df["benchmark_origin"].eq("defense_external_extension").all()

    # Check supported pairs (must have exactly 5 seeds)
    for d in DEFENSE_DATASETS:
        for m in SUPPORTED_MODELS:
            sub = df[(df["dataset"] == d) & (df["model"] == m)]
            assert len(sub) == 5, f"Expected 5 seeds for supported pair {d}/{m}, got {len(sub)}"
            observed_seeds = set(sub["seed"].astype(int))
            assert observed_seeds == EXPECTED_SEEDS, f"Seed mismatch for {d}/{m}: {observed_seeds}"
