from experiments.round6.policy import MAIN_REQUIREMENTS, evaluate_gate_v7


def _checks(value: bool) -> dict[str, bool]:
    checks = {key: value for key in MAIN_REQUIREMENTS}
    checks.update({
        "dataset_exact_recovery": value,
        "new_dataset_version_fully_retrained": False,
        "independent_benign_adjudication": False,
        "common_support_balance": False,
        "semantic_alignment_test": False,
        "same_corpus_retrieval_baseline": True,
    })
    return checks


def test_gate_m_does_not_depend_on_scam_gate():
    gate = evaluate_gate_v7(_checks(True))
    assert gate["gate_m_main_timestamp_gnn"] is True
    assert gate["gate_a_scam_graphrag"] is False
    assert gate["gate_b_full_cross_layer"] is False


def test_recovery_is_mandatory_for_gate_m():
    checks = _checks(True)
    checks["dataset_exact_recovery"] = False
    gate = evaluate_gate_v7(checks)
    assert gate["dataset_recovery_condition_satisfied"] is False
    assert gate["gate_m_main_timestamp_gnn"] is False

