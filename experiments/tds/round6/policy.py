"""Independent publication gates for the Round 6 final evidence package."""
from __future__ import annotations


MAIN_REQUIREMENTS = (
    "future_edge_audit_verified",
    "chronological_5seed_evaluation_complete",
    "tgn_complete",
    "tgat_complete",
    "fraud_specific_baseline_complete",
    "temperature_scaling_complete",
    "deep_ensemble_complete",
    "gate_required_model_comparisons_complete",
    "paired_bootstrap_complete",
    "randomization_analysis_complete",
    "reliability_figures_complete",
    "latency_scope_consistent",
    "title_claims_match_evidence",
    "publication_format_consistent",
)


def evaluate_gate_v7(checks: dict[str, bool]) -> dict[str, object]:
    """Evaluate Main, Scam, and Cross-Layer gates without cross-gate leakage."""
    recovery_ok = bool(
        checks.get("dataset_exact_recovery", False)
        or checks.get("new_dataset_version_fully_retrained", False)
    )
    main_ok = recovery_ok and all(bool(checks.get(key, False)) for key in MAIN_REQUIREMENTS)
    scam_ok = all(bool(checks.get(key, False)) for key in (
        "independent_benign_adjudication",
        "common_support_balance",
        "semantic_alignment_test",
        "same_corpus_retrieval_baseline",
    ))
    cross_layer_ok = scam_ok and all(bool(checks.get(key, False)) for key in (
        "real_wallet_transactions",
        "real_hash_timestamp_lineage",
        "scamwallet_onchain_v1",
        "dlg_gnn_5seed",
        "dlg_permutation_sanity",
        "cross_layer_complete_cases",
    ))
    return {
        "version": "publication-readiness-gate-v7.0",
        **checks,
        "dataset_recovery_condition_satisfied": recovery_ok,
        "gate_m_main_timestamp_gnn": main_ok,
        "gate_a_scam_graphrag": scam_ok,
        "gate_b_full_cross_layer": cross_layer_ok,
        "main_manuscript_draft_allowed": main_ok,
        "scam_claims_allowed": scam_ok,
        "full_cross_layer_claims_allowed": cross_layer_ok,
        "fail_closed": True,
    }

