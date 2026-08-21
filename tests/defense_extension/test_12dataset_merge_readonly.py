"""Test 12-dataset derived merge read-only integrity."""
from pathlib import Path
import pandas as pd
import pytest

FROZEN_RAW_SHA256 = "39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c"


def test_12dataset_view_provenance_and_round5_integrity():
    r5_path = Path("outputs/sci_round5_final/raw/benchmark_raw.csv")
    view_path = Path("outputs/sci_defense_extension/extended_analysis/benchmark_12dataset_view.csv")

    assert r5_path.exists()
    # Ensure Round 5 file was not modified during merge
    import hashlib
    r5_hash = hashlib.sha256(r5_path.read_bytes()).hexdigest()
    assert r5_hash == FROZEN_RAW_SHA256, "Round 5 raw was modified during 12-dataset merge!"

    if not view_path.exists():
        pytest.skip("12-dataset view not generated yet")

    df = pd.read_csv(view_path)
    assert not df.empty
    assert "benchmark_origin" in df.columns

    origins = set(df["benchmark_origin"].unique())
    assert origins == {"round5_primary", "defense_external_extension"}

    # Total datasets = 10 (round 5) + 2 (defense) = 12
    datasets = set(df["dataset"].unique())
    assert len(datasets) == 12
    assert "DARPA-TC-THEIA" in datasets
    assert "LANL-RedTeam" in datasets
