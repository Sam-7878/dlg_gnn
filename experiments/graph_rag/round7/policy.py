"""Fail-closed publication policy for Main Paper Gate v8."""
from __future__ import annotations


MAIN_REQUIREMENTS = (
    "dataset_provenance_complete",
    "chronological_split_verified",
    "future_edge_count_zero",
    "proposed_5seed_complete",
    "tgn_5seed_complete",
    "tgat_5seed_complete",
    "fraud_baseline_5seed_complete",
    "temperature_scaling_complete",
    "deep_ensemble_complete",
    "mc_dropout_complete",
    "raw_predictions_complete",
    "checkpoint_hashes_complete",
    "paired_bootstrap_complete",
    "class_stratified_bootstrap_complete",
    "randomization_analysis_complete",
    "temporal_slice_analysis_complete",
    "calibration_figure_complete",
    "latency_scope_consistent",
    "publication_format_consistent",
    "title_claims_match_evidence",
)


def evaluate_gate_v8(checks: dict[str, bool]) -> dict[str, object]:
    main_ok = all(bool(checks.get(key, False)) for key in MAIN_REQUIREMENTS)
    scam_ok = bool(
        checks.get("independent_double_human_annotations_ge_300", False)
        and checks.get("independent_benign_controls_sufficient", False)
    )
    cross_layer_ok = bool(
        scam_ok
        and checks.get("authorized_real_wallet_transaction_hashes", False)
        and checks.get("real_block_timestamps", False)
        and checks.get("sufficient_complete_cross_layer_cases", False)
    )
    return {
        "version": "publication-readiness-gate-v8.0",
        **checks,
        "gate_m_main_timestamp_gnn": main_ok,
        "gate_a_scam_graphrag": scam_ok,
        "gate_b_full_cross_layer": cross_layer_ok,
        "main_manuscript_finalization_allowed": main_ok,
        "scam_model_tuning_allowed": scam_ok,
        "cross_layer_claims_allowed": cross_layer_ok,
        "blocking_main_requirements": [key for key in MAIN_REQUIREMENTS if not checks.get(key, False)],
        "fail_closed": True,
    }

