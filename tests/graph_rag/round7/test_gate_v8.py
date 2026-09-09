from experiments.round7.policy import MAIN_REQUIREMENTS, evaluate_gate_v8


def test_gate_v8_requires_every_main_condition() -> None:
    checks = {key: True for key in MAIN_REQUIREMENTS}
    result = evaluate_gate_v8(checks)
    assert result["gate_m_main_timestamp_gnn"] is True
    checks["tgn_5seed_complete"] = False
    result = evaluate_gate_v8(checks)
    assert result["gate_m_main_timestamp_gnn"] is False
    assert result["blocking_main_requirements"] == ["tgn_5seed_complete"]


def test_scam_and_cross_layer_gates_remain_independent() -> None:
    checks = {key: True for key in MAIN_REQUIREMENTS}
    result = evaluate_gate_v8(checks)
    assert result["gate_m_main_timestamp_gnn"] is True
    assert result["gate_a_scam_graphrag"] is False
    assert result["gate_b_full_cross_layer"] is False

