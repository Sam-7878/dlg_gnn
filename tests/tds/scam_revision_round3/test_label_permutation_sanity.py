import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/graphrag/scam_revision_round3"


def test_permutation_outcome_is_propagated_to_fail_closed_gate():
    campaign = pd.read_csv(RESULTS / "label_permutation.csv")
    gog = pd.read_csv(RESULTS / "gog_dlg_permutation_audit.csv")
    gate = json.loads((RESULTS / "paper_ready_gate_v4.json").read_text())
    campaign_ok = (
        abs(campaign.roc_auc.mean() - 0.5) <= 0.10
        and abs(campaign.auc_pr.mean() - campaign.positive_prevalence.iloc[0]) <= 0.10
    )
    shuffled = gog[gog["mode"] == "permuted_train_labels"]
    gog_ok = (
        abs(shuffled.roc_auc.mean() - 0.5) <= 0.10
        and abs(shuffled.auc_pr.mean() - shuffled.positive_prevalence.iloc[0]) <= 0.10
    )
    expected = bool(campaign_ok and gog_ok)
    assert gate["checks"]["dlg_permutation_test_pass"] is expected
    if not expected:
        assert gate["paper_ready"] is False
