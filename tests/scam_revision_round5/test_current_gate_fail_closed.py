import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_current_gate_records_missing_training_data_without_fake_metrics():
    gate = json.loads((ROOT / "results/paper_ready_gate_v6.json").read_text())
    assert not gate["gate_m_main_timestamp_gnn"]
    assert gate["deep_ensemble_or_uncertainty_baseline_complete"]
    assert not gate["temperature_scaling_complete"]
    assert not gate["temporal_baselines_complete"]
    assert not gate["fraud_specific_baseline_complete"]
    assert gate["data_availability"]["held_out_fraud_positives"] == 107
