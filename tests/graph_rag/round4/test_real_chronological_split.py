import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_real_chronological_split():
    manifest = json.loads((ROOT / "data/benchmark/gog_scimain_v1/real_dataset_manifest.json").read_text())
    split = manifest["split"]
    assert split["train"]["end_time"] <= split["validation"]["start_time"]
    assert split["validation"]["end_time"] <= split["test"]["start_time"]
    for name in ("train", "validation", "test"):
        assert split[name]["n_positive"] > 0
        assert split[name]["n_negative"] > 0

