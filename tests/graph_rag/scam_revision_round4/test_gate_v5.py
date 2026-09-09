from graphrag.scam_revision.round4_final_evidence import evaluate_gate_v5


GATE_A = {
    "independent_benign_adjudication_pass": True,
    "common_support_balance_pass": True,
    "degree_shortcut_not_near_perfect": True,
    "fixed_retrieval_gold": True,
    "global_retrieval_baselines_complete": True,
    "semantic_alignment_test_complete": True,
    "cross_source_two_class_adjudicated": True,
    "claim_metric_consistency": True,
}
GATE_B = {
    "real_scam_wallet_transactions_available": True,
    "real_transaction_hash_timestamp_lineage": True,
    "onchain_coverage_sufficient": True,
    "real_dlg_gnn_checkpoint_5seed": True,
    "dlg_permutation_sanity_pass": True,
    "cross_layer_complete_sample_support": True,
}


def test_gate_a_is_independent_but_paper_ready_requires_gate_b():
    graph_only = evaluate_gate_v5({**GATE_A, **{key: False for key in GATE_B}})
    assert graph_only["gate_a_graphrag_paper"]
    assert not graph_only["gate_b_full_cross_layer"]
    assert graph_only["paper_ready"]
    full = evaluate_gate_v5({**GATE_A, **GATE_B})
    assert full["gate_b_full_cross_layer"] and full["paper_ready"]
