"""Independent publication decision gate for Round 5."""
from __future__ import annotations

from collections.abc import Mapping


GATE_M_KEYS = (
    "temporal_baselines_complete",
    "fraud_specific_baseline_complete",
    "temperature_scaling_complete",
    "deep_ensemble_or_uncertainty_baseline_complete",
    "chronological_5seed_evaluation_complete",
    "paired_bootstrap_complete",
    "permutation_randomization_comparison_complete",
    "reliability_figures_complete",
    "title_claims_match_evidence",
    "publication_format_consistent",
)

GATE_A_KEYS = (
    "independent_benign_adjudication",
    "common_support_balance",
    "shortcut_resistance",
    "semantic_alignment_test",
    "same_corpus_retrieval_baseline",
    "adjudicated_two_class_generalization",
)

GATE_B_EXTRA_KEYS = (
    "real_wallet_transactions",
    "real_hash_timestamp_lineage",
    "scamwallet_onchain_v1",
    "dlg_gnn_5seed",
    "dlg_permutation_sanity",
    "cross_layer_complete_cases",
)


def evaluate_gate_v6(checks: Mapping[str, object]) -> dict[str, object]:
    normalized = {
        key: bool(checks.get(key, False))
        for key in GATE_M_KEYS + GATE_A_KEYS + GATE_B_EXTRA_KEYS
    }
    gate_m = all(normalized[key] for key in GATE_M_KEYS)
    gate_a = all(normalized[key] for key in GATE_A_KEYS)
    gate_b = gate_a and all(normalized[key] for key in GATE_B_EXTRA_KEYS)
    return {
        "version": "publication-readiness-gate-v6.0",
        **normalized,
        "gate_m_main_timestamp_gnn": gate_m,
        "gate_a_scam_graphrag": gate_a,
        "gate_b_full_cross_layer": gate_b,
        "main_manuscript_draft_allowed": gate_m,
        "scam_claims_allowed": gate_a,
        "full_cross_layer_claims_allowed": gate_b,
        "fail_closed": True,
    }
