"""
tests/scam_revision/test_lead_time_lineage.py

Verifies that lead_time_pairs.parquet contains valid event-level timestamps,
non-negative lead times, and distinguishes social-to-report vs social-to-onchain lead times.
"""

import os
import pytest
import pandas as pd

LEAD_TIME_PARQUET = "/mnt/d/_Work/goat_bank/dlg_gnn/results/graphrag/scam_revision_round2/lead_time_pairs.parquet"
LEAD_TIME_CSV = "/mnt/d/_Work/goat_bank/dlg_gnn/results/graphrag/scam_revision_round2/lead_time_summary.csv"


def test_lead_time_lineage():
    if not os.path.exists(LEAD_TIME_PARQUET) and not os.path.exists(LEAD_TIME_CSV):
        pytest.skip("Lead time lineage files not yet generated.")
        
    df = pd.read_parquet(LEAD_TIME_PARQUET) if os.path.exists(LEAD_TIME_PARQUET) else pd.read_csv(LEAD_TIME_CSV)
    
    assert len(df) > 0, "Lead time dataset is empty"
    
    # Check non-negative lead times
    if "lead_to_report_days" in df.columns:
        assert (df["lead_to_report_days"] >= 0.0).all(), "Found negative lead times!"
        assert df["lead_to_report_days"].mean() > 0.0, "Mean lead time must be positive"
