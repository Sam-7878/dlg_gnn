import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_recorded_timestamp_required():
    manifest = json.loads((ROOT / "data/benchmark/gog_scimain_v1/real_dataset_manifest.json").read_text())
    assert manifest["timestamp_source"] == "recorded_transaction_timestamp"
    assert manifest["timezone"].startswith("UTC")
    assert manifest["paper_eligible"] is True

