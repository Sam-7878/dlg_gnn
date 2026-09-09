import json
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def test_future_edge_zero():
    manifest = json.loads((ROOT / "data/benchmark/gog_scimain_v1/real_dataset_manifest.json").read_text())
    audit = pd.read_csv(ROOT / "data/benchmark/gog_scimain_v1/future_edge_audit.csv")
    assert manifest["future_edge_count"] == 0
    assert len(audit) >= 20
    assert (audit.future_edge_count == 0).all()
    assert (audit.max_input_edge_timestamp <= audit.event_timestamp).all()

