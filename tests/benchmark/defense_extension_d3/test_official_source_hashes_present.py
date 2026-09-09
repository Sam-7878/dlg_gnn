import json
from pathlib import Path


def test_available_official_source_hashes_present_and_raw_gate_fails_closed(d3_gate):
    # This test covers the preserved pre-acquisition D3 gate, not the later
    # real-source CSV that now occupies source_audit/darpa_raw_manifest.csv.
    manifest = Path("outputs/sci_defense_extension_real/manifests/darpa_real_manifest.json")
    rows = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    available = [row for row in rows if row["available"]]
    assert available and all(len(row["sha256"]) == 64 for row in available)
    assert d3_gate["darpa"]["ground_truth_available"]
    assert d3_gate["darpa"]["schema_available"]
    assert not d3_gate["darpa"]["raw_available"]
    assert d3_gate["decision"] == "PAPER_READY_10_DATASET_ONLY"
