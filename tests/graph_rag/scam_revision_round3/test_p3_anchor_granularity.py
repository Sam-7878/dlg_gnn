from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/graphrag/scam_revision_round3"


def test_p3_strong_and_weak_are_separated_at_anchor_granularity():
    manifest = pd.read_parquet(RESULTS / "label_manifest_v2.parquet")
    campaigns = manifest[manifest.entity_type == "campaign"]
    strong = campaigns[campaigns.label_tier == "P3-Strong"]
    weak = campaigns[campaigns.label_tier == "P3-Weak"]
    assert len(strong) > 0 and len(weak) > 0
    assert set(strong.anchor_type) <= {
        "exact_wallet", "exact_full_url", "exact_shared_path", "dedicated_malicious_host"
    }
    assert strong.main_eligible.astype(bool).all()
    assert not weak.main_eligible.astype(bool).any()
    assert set(weak.anchor_type) == {"shared_platform_root_only"}
