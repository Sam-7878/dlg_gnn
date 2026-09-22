import pandas as pd

from graphrag.scam_revision.round4_final_evidence import gog_consistency


def test_zero_exact_match_forces_zero_real_campaigns():
    frame = pd.DataFrame([{
        "campaign_id": "c1", "registry_wallet": "", "wallet_chain": "chain_unknown",
        "gog_entity_id": "0xabc", "match_type": "ccc_proxy", "match_key": "0xabc",
        "exact_match": False, "legacy_proxy": True, "source_file": "x", "source_row": "1",
    }])
    result = gog_consistency(frame)
    assert result["exact_match_count"] == 0
    assert result["campaigns_with_real_gog_wallet_evidence"] == 0
    assert result["pass"]
