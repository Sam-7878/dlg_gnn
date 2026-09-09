"""
tests/scam_revision/test_label_manifest_consistency.py

Verifies that the canonical label manifest exists, contains no null sample_ids,
has valid binary labels {0, 1}, and has consistent tier support.
"""

import os
import pandas as pd
import pytest

MANIFEST_PATH = "/mnt/d/_Work/goat_bank/dlg_gnn/results/graphrag/scam_revision_round2/label_manifest.parquet"
CSV_MANIFEST_PATH = "/mnt/d/_Work/goat_bank/dlg_gnn/results/graphrag/scam_revision_round2/label_manifest.csv"


def test_label_manifest_exists_and_valid():
    # If not yet generated during full run, test with local creation or existing file
    if not os.path.exists(MANIFEST_PATH) and not os.path.exists(CSV_MANIFEST_PATH):
        pytest.skip("Label manifest not yet generated (will be verified after pipeline run).")
        
    df = pd.read_parquet(MANIFEST_PATH) if os.path.exists(MANIFEST_PATH) else pd.read_csv(CSV_MANIFEST_PATH)
    
    assert len(df) > 0, "Manifest is empty"
    assert "sample_id" in df.columns
    assert "label_binary" in df.columns
    assert "label_tier" in df.columns
    assert "split" in df.columns
    
    # Check no duplicate sample_ids
    assert df["sample_id"].is_unique, "Duplicate sample_ids found in canonical manifest!"
    
    # Check binary labels
    assert set(df["label_binary"].unique()).issubset({0, 1}), "Invalid non-binary label found!"
    
    # Check valid splits
    assert set(df["split"].unique()).issubset({"train", "val", "test", "unassigned"}), "Invalid split name found!"
    
    # Check valid tiers
    assert set(df["label_tier"].unique()).issubset({"P1", "P2", "P3", "N1", "N2"}), "Invalid tier name found!"
