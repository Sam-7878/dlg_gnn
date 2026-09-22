"""
tests/scam_revision/test_claim_metric_consistency.py

Phase 22: Automated Claim vs Metric Consistency Checker
Ensures that narrative conclusions in reports strictly match mathematical metric signs.
If a fusion model or retrieval hop decreases performance, reports cannot claim an 'improvement'.
"""

import os
import pytest
import pandas as pd

REPORT_PATH = "/mnt/d/_Work/goat_bank/dlg_gnn/reports/graphrag/scam_revision_round2/final_validation_report.md"
MAIN_METRICS_PATH = "/mnt/d/_Work/goat_bank/dlg_gnn/results/graphrag/scam_revision_round2/main_detection.csv"
RETRIEVAL_METRICS_PATH = "/mnt/d/_Work/goat_bank/dlg_gnn/results/graphrag/scam_revision_round2/retrieval_metrics.csv"


def test_claim_metric_consistency():
    if not os.path.exists(REPORT_PATH) or not os.path.exists(MAIN_METRICS_PATH):
        pytest.skip("Round 2 report or metrics not yet generated.")
        
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report_text = f.read().lower()
        
    df_main = pd.read_csv(MAIN_METRICS_PATH)
    
    # 1. Check Fusion vs GNN delta sign
    gnn_auc = df_main["gnn_auc_pr"].mean()
    fusion_auc = df_main["uncertainty_auc_pr"].mean()
    delta = fusion_auc - gnn_auc
    
    if delta < 0:
        assert "uncertainty fusion improves dlg-gnn by +" not in report_text, (
            f"Contradiction: Metric delta is negative ({delta:.4f}), but report claimed positive improvement!"
        )
        assert "positive fusion-gain" not in report_text

    # 2. Check Retrieval MRR delta sign
    if os.path.exists(RETRIEVAL_METRICS_PATH):
        df_ret = pd.read_csv(RETRIEVAL_METRICS_PATH)
        mrr_0 = df_ret.loc[df_ret["hop"] == 0, "mrr_mean"].values[0]
        mrr_2 = df_ret.loc[df_ret["hop"] == 2, "mrr_mean"].values[0]
        
        if mrr_2 <= mrr_0:
            assert "statistically significant mrr gain" not in report_text, (
                f"Contradiction: 2-hop MRR ({mrr_2:.4f}) <= 0-hop MRR ({mrr_0:.4f}), but report claimed MRR gain!"
            )
