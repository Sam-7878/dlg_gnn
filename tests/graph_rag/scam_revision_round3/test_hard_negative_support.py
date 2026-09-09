import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/graphrag/scam_revision_round3"


def test_hard_negative_count_and_verification_boundary():
    manifest = pd.read_parquet(RESULTS / "label_manifest_v2.parquet")
    gate = json.loads((RESULTS / "paper_ready_gate_v4.json").read_text())
    negatives = manifest[(manifest.entity_type == "campaign") & (manifest.label_tier == "N1")]
    assert len(negatives) >= 500
    assert negatives.sample_id.is_unique
    assert (negatives.label.astype(int) == 0).all()
    assert negatives.negative_verification.str.contains("time/feature-matched").all()
    independently_verified = ~negatives.negative_verification.str.contains("not manually adjudicated")
    assert gate["checks"]["hard_negatives_independently_verified"] is bool(independently_verified.all())
    if not independently_verified.all():
        assert gate["paper_ready"] is False
