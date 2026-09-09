import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_paper_ready_gate_v2():
    gate = json.loads((ROOT / "results/graphrag/round_4/paper_ready_gate.json").read_text())
    assert gate["gate_version"] == "round4-paper-ready-v2.0"
    assert gate["paper_ready"] is True
    assert all(gate["checks"].values())
    assert gate["controlled_track_promoted"] is False

