"""
tests/scam_revision/test_split_entity_disjointness.py

Verifies that disjoint splits have zero entity overlap between train and test sets.
"""

import os
import pytest
import pandas as pd

SPLITS_DIR = "/mnt/d/_Work/goat_bank/dlg_gnn/results/graphrag/scam_revision_round2/split_manifests"
MANIFEST_PATH = "/mnt/d/_Work/goat_bank/dlg_gnn/results/graphrag/scam_revision_round2/label_manifest.parquet"
CSV_MANIFEST_PATH = "/mnt/d/_Work/goat_bank/dlg_gnn/results/graphrag/scam_revision_round2/label_manifest.csv"


def test_split_entity_disjointness():
    if not os.path.exists(SPLITS_DIR):
        pytest.skip("Split manifests not yet generated.")
        
    df = pd.read_parquet(MANIFEST_PATH) if os.path.exists(MANIFEST_PATH) else pd.read_csv(CSV_MANIFEST_PATH)
    
    # 1. Campaign Disjoint
    c_split_file = os.path.join(SPLITS_DIR, "campaign_disjoint_test_ids.txt")
    if os.path.exists(c_split_file):
        with open(c_split_file) as f:
            test_ids = set(line.strip() for line in f if line.strip())
        train_ids = set(df[df["split"] == "train"]["sample_id"])
        overlap = test_ids & train_ids
        assert len(overlap) == 0, f"Campaign disjoint leakage: {len(overlap)} overlapping IDs found!"

    # 2. Wallet Disjoint
    w_split_file = os.path.join(SPLITS_DIR, "wallet_disjoint_test_ids.txt")
    if os.path.exists(w_split_file):
        with open(w_split_file) as f:
            test_ids = set(line.strip() for line in f if line.strip())
        train_wallets = set(df[df["split"] == "train"]["wallet"].dropna()) - {""}
        test_wallets = set(df[df["sample_id"].isin(test_ids)]["wallet"].dropna()) - {""}
        overlap = test_wallets & train_wallets
        assert len(overlap) == 0, f"Wallet disjoint leakage: {len(overlap)} overlapping wallets found!"

    # 3. Domain Disjoint
    d_split_file = os.path.join(SPLITS_DIR, "domain_disjoint_test_ids.txt")
    if os.path.exists(d_split_file):
        with open(d_split_file) as f:
            test_ids = set(line.strip() for line in f if line.strip())
        train_domains = set(df[df["split"] == "train"]["domain"].dropna()) - {""}
        test_domains = set(df[df["sample_id"].isin(test_ids)]["domain"].dropna()) - {""}
        overlap = test_domains & train_domains
        assert len(overlap) == 0, f"Domain disjoint leakage: {len(overlap)} overlapping domains found!"
