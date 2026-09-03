from experiments.round5.policy import GATE_A_KEYS, GATE_B_EXTRA_KEYS, GATE_M_KEYS, evaluate_gate_v6


def test_main_gate_is_independent_from_scam_and_cross_layer_gates():
    checks = {key: True for key in GATE_M_KEYS}
    checks.update({key: False for key in GATE_A_KEYS + GATE_B_EXTRA_KEYS})
    result = evaluate_gate_v6(checks)
    assert result["gate_m_main_timestamp_gnn"]
    assert result["main_manuscript_draft_allowed"]
    assert not result["gate_a_scam_graphrag"]
    assert not result["gate_b_full_cross_layer"]
