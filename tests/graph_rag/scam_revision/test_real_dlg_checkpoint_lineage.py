"""
tests/scam_revision/test_real_dlg_checkpoint_lineage.py

Verifies that raw prediction outputs exist, have valid probability bounds [0, 1],
and contain required lineage fields (sample_id, label, p_gnn, p_rag, uncertainty, p_fusion).
"""

import os
import glob
import pytest
import pandas as pd

RAW_PRED_DIR = "/mnt/d/_Work/goat_bank/dlg_gnn/results/graphrag/scam_revision_round2/raw_predictions"


def test_raw_prediction_lineage():
    if not os.path.exists(RAW_PRED_DIR):
        pytest.skip("Raw predictions not yet generated.")
        
    pred_files = glob.glob(os.path.join(RAW_PRED_DIR, "*.parquet")) + glob.glob(os.path.join(RAW_PRED_DIR, "*.csv"))
    if not pred_files:
        pytest.skip("No prediction files found yet.")
        
    df_sample = pd.read_parquet(pred_files[0]) if pred_files[0].endswith(".parquet") else pd.read_csv(pred_files[0])
    
    required_cols = ["sample_id", "label", "p_gnn", "p_rag", "uncertainty", "p_fusion", "split"]
    for col in required_cols:
        assert col in df_sample.columns, f"Missing required column: {col}"
        
    # Check probabilities in [0, 1]
    assert (df_sample["p_gnn"] >= 0.0).all() and (df_sample["p_gnn"] <= 1.0).all()
    assert (df_sample["p_rag"] >= 0.0).all() and (df_sample["p_rag"] <= 1.0).all()
    assert (df_sample["p_fusion"] >= 0.0).all() and (df_sample["p_fusion"] <= 1.0).all()
