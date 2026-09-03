import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/graphrag/scam_revision_round3"


def test_onchain_claim_is_fail_closed_when_transaction_lineage_is_absent():
    lead = pd.read_parquet(RESULTS / "lead_time_pairs_real.parquet")
    summary = pd.read_csv(RESULTS / "lead_time_real_summary.csv")
    gate = json.loads((RESULTS / "paper_ready_gate_v4.json").read_text())
    eligible = lead[lead.real_onchain_time.astype(bool)]
    if len(eligible):
        assert eligible.transaction_hash_or_event_id.fillna("").ne("").all()
        assert eligible.first_observed_transaction_time.notna().all()
    else:
        row = summary[summary.metric == "social_to_onchain"].iloc[0]
        assert row.eligible_n == 0 and not bool(row.paper_eligible)
        assert gate["checks"]["real_onchain_event_lineage_verified"] is False
        assert gate["paper_ready"] is False
