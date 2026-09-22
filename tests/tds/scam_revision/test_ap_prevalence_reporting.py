"""
tests/scam_revision/test_ap_prevalence_reporting.py

Verifies that all detection result artifacts report positive prevalence
and compute normalized Average Precision (AP) lift alongside raw AUC-PR.
"""

import os
import pytest
import pandas as pd

MAIN_DETECTION_CSV = "/mnt/d/_Work/goat_bank/dlg_gnn/results/graphrag/scam_revision_round2/main_detection.csv"
TABLE6_CSV = "/mnt/d/_Work/goat_bank/dlg_gnn/tables/graphrag/scam_revision_round2/table6_main_detection.csv"


def test_ap_prevalence_reporting():
    if not os.path.exists(MAIN_DETECTION_CSV):
        pytest.skip("Main detection CSV not yet generated.")
        
    df = pd.read_csv(MAIN_DETECTION_CSV)
    
    # Check that AP lift and prevalence are tracked in raw seed metrics
    assert "positive_prevalence" in df.columns
    assert "gnn_ap_lift" in df.columns
    assert "uncertainty_ap_lift" in df.columns
    
    # Check that table6 also reports AP lift
    if os.path.exists(TABLE6_CSV):
        df_t6 = pd.read_csv(TABLE6_CSV)
        assert "Model" in df_t6.columns
        assert "AP Lift" in df_t6.columns
