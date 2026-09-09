"""
tests/scam_revision/test_bridge_manifest_consistency.py

Verifies that bridge_manifest.csv contains valid counts and non-zero exact bridges.
"""

import os
import pytest
import pandas as pd

BRIDGE_MANIFEST_PATH = "/mnt/d/_Work/goat_bank/dlg_gnn/results/graphrag/scam_revision_round2/bridge_manifest.csv"


def test_bridge_manifest_consistency():
    if not os.path.exists(BRIDGE_MANIFEST_PATH):
        pytest.skip("Bridge manifest not yet generated.")
        
    df = pd.read_csv(BRIDGE_MANIFEST_PATH)
    assert len(df) == 1, "Bridge manifest should have 1 summary record"
    row = df.iloc[0]
    
    # Assert non-zero counts
    assert row["cst_unique_wallets"] > 1000
    assert row["cst_unique_domains"] > 1000
    assert row["cst_domain_wallet_links"] > 1000
    assert row["csdb_unique_wallets"] > 1000
    assert row["csdb_unique_domains"] > 1000
    assert row["total_constructed_bridges"] > 10000
