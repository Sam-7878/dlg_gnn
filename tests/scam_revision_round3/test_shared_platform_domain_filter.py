import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/graphrag/scam_revision_round3"


def test_shared_platform_roots_never_become_strong_campaign_anchors():
    manifest = pd.read_parquet(RESULTS / "label_manifest_v2.parquet")
    audit = pd.read_csv(RESULTS / "shared_platform_domain_audit.csv")
    shared = set(audit.loc[audit.is_shared_platform, "registered_domain"])
    strong = manifest[(manifest.entity_type == "campaign") & (manifest.label_tier == "P3-Strong")]
    assert not strong.anchor_value.isin(shared).any()
    assert not audit.loc[audit.is_shared_platform, "eligible_as_root_anchor"].astype(bool).any()
    gate = json.loads((RESULTS / "paper_ready_gate_v4.json").read_text())
    assert gate["checks"]["shared_platform_domain_contamination_removed"] is True
