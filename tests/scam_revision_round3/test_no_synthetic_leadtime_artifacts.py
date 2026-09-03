from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/graphrag/scam_revision_round3"


def test_lead_time_rows_have_no_round2_placeholder_patterns():
    lead = pd.read_parquet(RESULTS / "lead_time_pairs_real.parquet")
    assert not lead.campaign_id.str.match(r"^ccc:1\d{3}$").all()
    nonempty_wallets = lead.wallet.fillna("").ne("")
    assert not lead.loc[nonempty_wallets, "wallet"].str.match(r"^0x0{30,}[0-9a-f]{1,10}$").any()
    times = np.sort(lead.social_signal_time.dropna().astype("int64").unique())
    if len(times) > 20:
        assert len(np.unique(np.diff(times))) > 2
    real_chain = lead[lead.real_onchain_time.astype(bool)]
    if len(real_chain) > 20:
        offsets = real_chain.first_observed_transaction_time - real_chain.social_signal_time
        assert offsets.nunique() > 2
