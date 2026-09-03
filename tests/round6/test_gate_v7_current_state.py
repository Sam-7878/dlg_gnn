import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_current_gate_is_fail_closed_when_frozen_data_is_missing():
    path = ROOT / "results" / "paper_ready_gate_v7.json"
    if not path.is_file():
        return
    gate = json.loads(path.read_text())
    assert gate["dataset_exact_recovery"] is False
    assert gate["future_edge_audit_verified"] is False
    assert gate["available_panel_bootstrap_complete"] is True
    assert gate["gate_required_model_comparisons_complete"] is False
    assert gate["gate_m_main_timestamp_gnn"] is False
    assert gate["gate_a_scam_graphrag"] is False
    assert gate["gate_b_full_cross_layer"] is False
