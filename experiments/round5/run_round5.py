"""Run publication-readiness analyses without manufacturing missing evidence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.round5.analysis import (
    build_calibration_results, build_statistical_comparisons,
    build_temporal_shift_analysis, freeze_annotation_package,
    generate_figures, write_text,
)
from experiments.round5.policy import evaluate_gate_v6


ROUND4_MAIN = ROOT / "results" / "graphrag" / "round_4"
ROUND4_SCAM = ROOT / "results" / "graphrag" / "scam_revision_round4"
SCAM_RESULTS = ROOT / "results" / "graphrag" / "scam_revision_round5"
SCAM_REPORTS = ROOT / "reports" / "graphrag" / "scam_revision_round5"
MAIN_RESULTS = ROOT / "results" / "main_final"
MAIN_FIGURES = ROOT / "figures" / "main_final"
MAIN_REPORTS = ROOT / "reports" / "main_final"
GATE_PATH = ROOT / "results" / "paper_ready_gate_v6.json"
DATASET = ROOT / "data" / "benchmark" / "gog_scimain_v1"
UPSTREAM = Path("/mnt/d/_Work/_data/GoG_sci_v2")
PAPER = ROOT / "docs" / "papers" / "_43_01_GraphRAG" / "graphrag.tex"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_blocked_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_available = (DATASET / "graph.pt").is_file() and (DATASET / "transactions.parquet").is_file()
    reason = (
        "training not executed: frozen GoG-SCIMain-v1 packed dataset is absent; "
        "upstream /mnt/d/_Work/_data/GoG_sci_v2 is also absent"
    )
    temporal = pd.DataFrame([{
        "method": method,
        "implementation": implementation,
        "evaluation_split": "fixed_pooled_temporal_holdout_70_15_15",
        "seeds_required": "7;17;27;37;47",
        "auc_pr": np.nan, "auc_roc": np.nan, "f1": np.nan,
        "brier": np.nan, "ece": np.nan, "nll": np.nan,
        "status": "READY_NOT_RUN" if data_available else "BLOCKED_MISSING_FROZEN_DATASET",
        "reason": "runner required" if data_available else reason,
    } for method, implementation in (
        ("TGAT-style", "experiments.round5.models.TGATBaseline"),
        ("TGN-style event memory", "experiments.round5.models.TemporalMemoryBaseline"),
    )])
    fraud = pd.DataFrame([{
        "method": "FraudSAGE (CARE-oriented equivalent)",
        "implementation": "experiments.round5.models.FraudSAGEBaseline",
        "care_gnn_exact_reproduction": False,
        "justification": "GoG-SCIMain-v1 has no heterogeneous relation types required by CARE-GNN",
        "evaluation_split": "fixed_pooled_temporal_holdout_70_15_15",
        "seeds_required": "7;17;27;37;47",
        "auc_pr": np.nan, "auc_roc": np.nan, "f1": np.nan,
        "brier": np.nan, "ece": np.nan, "nll": np.nan,
        "status": "READY_NOT_RUN" if data_available else "BLOCKED_MISSING_FROZEN_DATASET",
        "reason": "runner required" if data_available else reason,
    }])
    temporal.to_csv(MAIN_RESULTS / "temporal_baselines.csv", index=False)
    fraud.to_csv(MAIN_RESULTS / "fraud_specific_baseline.csv", index=False)
    return temporal, fraud


def paper_checks() -> tuple[bool, bool, str]:
    source = PAPER.read_text(encoding="utf-8")
    title_line = next(line for line in source.splitlines() if line.startswith("\\title{"))
    forbidden = ("Privacy-Preserving", "GraphRAG", "Cross-Layer", "Early-Warning")
    title_ok = not any(term.lower() in title_line.lower() for term in forbidden)
    ieee_active = "\\documentclass[10pt,journal]{IEEEtran}" in source
    format_ok = ieee_active and "\\bibliographystyle{IEEEtran}" in source and "\\bibliographystyle{elsarticle-num}" not in source
    return title_ok, format_ok, title_line


def write_scam_status(annotation: dict, gate_v5: dict) -> None:
    SCAM_REPORTS.mkdir(parents=True, exist_ok=True)
    write_text(
        SCAM_REPORTS / "evidence_status.md",
        f"""# Auxiliary Scam-Campaign Evidence Status

The branch is frozen at the audited Round 4 evidence boundary. Exact registry-wallet/GoG matches are
{gate_v5['details']['gog']['exact_match_count']}; real transaction rows are
{gate_v5['details']['onchain']['transaction_rows']}. Same-corpus retrieval remains a
relation-aware reachability diagnostic. No campaign-detection, independent semantic superiority,
early-warning, privacy-preserving, or full cross-layer claim is enabled.
""",
    )
    write_text(
        SCAM_REPORTS / "annotation_status.md",
        f"""# Human Annotation Status

Package `{annotation['annotation_package_version']}` is frozen with **{annotation['sample_count']}**
candidates and SHA-256 `{annotation['package_sha256']}`. It contains no model score or prediction.
Double-annotated samples: **0**; consensus BENIGN: **0**. Gate A remains closed until at least 300
samples are independently double annotated, Cohen's kappa is reported, and sufficient BENIGN/BENIGN
final-BENIGN controls remain for held-out evaluation.
""",
    )
    write_text(
        SCAM_REPORTS / "onchain_source_status.md",
        """# Real Transaction Source Status

No local transaction-hash/block-timestamp archive for the registry wallets and no authorized
explorer/RPC source is configured. The local static GoG archive is not a substitute. ScamWallet-
OnChain-v1, temporal DLG-GNN, fusion, and lead-time loops remain stopped. No synthetic transactions,
proxy timestamps, fuzzy wallet matches, or default DLG scores were introduced.
""",
    )


def run() -> dict[str, object]:
    for path in (SCAM_RESULTS, SCAM_REPORTS, MAIN_RESULTS, MAIN_FIGURES, MAIN_REPORTS):
        path.mkdir(parents=True, exist_ok=True)

    annotation = freeze_annotation_package(
        ROUND4_SCAM / "benign_annotation_candidates.csv", SCAM_RESULTS,
    )
    gate_v5 = _json(ROUND4_SCAM / "paper_ready_gate_v5.json")
    write_scam_status(annotation, gate_v5)

    manifest = _json(ROUND4_MAIN / "real_dataset_manifest.json")
    calibration, ensemble_predictions = build_calibration_results(
        ROUND4_MAIN / "raw_predictions", MAIN_RESULTS,
    )
    statistics = build_statistical_comparisons(
        ROUND4_MAIN / "raw_predictions", ensemble_predictions, MAIN_RESULTS, 10_000,
    )
    temporal_shift = build_temporal_shift_analysis(
        manifest, ensemble_predictions, MAIN_RESULTS, bins=6,
    )
    figures = generate_figures(
        manifest, ROUND4_MAIN / "raw_predictions", ensemble_predictions,
        temporal_shift, ROUND4_MAIN / "mc_sensitivity.csv", MAIN_FIGURES, MAIN_RESULTS,
    )
    temporal_baselines, fraud_baseline = write_blocked_baselines()

    title_ok, format_ok, title_line = paper_checks()
    available_stats_complete = bool((statistics.status == "COMPLETE").sum() == 2)
    deep_complete = bool(
        ((calibration.method == "Deep Ensemble") & (calibration.n_models == 5)
         & (calibration.status == "COMPLETE_HELD_OUT_TEST")).any()
    )
    five_seed_complete = all(
        (ROUND4_MAIN / "raw_predictions" / f"seed{seed}_T1.csv").is_file()
        and (ROUND4_MAIN / "raw_predictions" / f"seed{seed}_T10.csv").is_file()
        and (ROUND4_MAIN / "real_checkpoints" / f"seed{seed}.pt").is_file()
        for seed in (7, 17, 27, 37, 47)
    )
    checks = {
        "temporal_baselines_complete": temporal_baselines.status.eq("COMPLETE").sum() >= 2,
        "fraud_specific_baseline_complete": fraud_baseline.status.eq("COMPLETE").any(),
        "temperature_scaling_complete": calibration.loc[
            calibration.method == "Temperature-Scaled GNN", "status"
        ].eq("COMPLETE_HELD_OUT_TEST").any(),
        "deep_ensemble_or_uncertainty_baseline_complete": deep_complete,
        "chronological_5seed_evaluation_complete": five_seed_complete,
        # Two required comparisons remain blocked, so the full statistical package is incomplete.
        "paired_bootstrap_complete": bool((statistics.status == "COMPLETE").all()),
        "permutation_randomization_comparison_complete": available_stats_complete,
        "reliability_figures_complete": figures["reliability_comparison_complete"],
        "title_claims_match_evidence": title_ok,
        "publication_format_consistent": format_ok,
        "independent_benign_adjudication": False,
        "common_support_balance": False,
        "shortcut_resistance": bool(gate_v5.get("degree_shortcut_not_near_perfect", False)),
        "semantic_alignment_test": False,
        "same_corpus_retrieval_baseline": bool(gate_v5.get("global_retrieval_baselines_complete", False)),
        "adjudicated_two_class_generalization": False,
        "real_wallet_transactions": False,
        "real_hash_timestamp_lineage": False,
        "scamwallet_onchain_v1": False,
        "dlg_gnn_5seed": False,
        "dlg_permutation_sanity": False,
        "cross_layer_complete_cases": False,
    }
    gate = evaluate_gate_v6(checks)
    gate["data_availability"] = {
        "packed_dataset_present": (DATASET / "graph.pt").is_file(),
        "transaction_metadata_present": (DATASET / "transactions.parquet").is_file(),
        "future_edge_audit_present": (DATASET / "future_edge_audit.csv").is_file(),
        "upstream_derivative_present": UPSTREAM.is_dir(),
        "saved_five_seed_checkpoints_present": five_seed_complete,
        "held_out_prediction_events": int(len(ensemble_predictions)),
        "held_out_fraud_positives": int(ensemble_predictions.label.sum()),
    }
    gate["partial_completion"] = {
        "deep_ensemble_complete": deep_complete,
        "available_event_level_statistics_complete": available_stats_complete,
        "temporal_slice_analysis_complete": True,
        "figures_generated": [key for key, value in figures.items() if value],
        "annotation_package_frozen": True,
        "paper_title": title_line,
    }
    GATE_PATH.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    deep_row = calibration[calibration.method == "Deep Ensemble"].iloc[0]
    mc_row = calibration[calibration.method == "MC Dropout Ensemble T=10"].iloc[0]
    write_text(
        MAIN_REPORTS / "publication_readiness_report.md",
        f"""# Main Timestamp-GNN Publication Readiness

Gate M: **{gate['gate_m_main_timestamp_gnn']}**. The existing five frozen checkpoints and 3,648
aligned held-out predictions support a valid five-model deep ensemble, 10,000-resample paired
bootstrap/randomization comparisons, six sequential test slices, and four generated figures.

Deep Ensemble: AUC-PR **{deep_row.auc_pr:.4f}**, ECE **{deep_row.ece:.4f}**, NLL
**{deep_row.nll:.4f}**. MC Dropout Ensemble T=10: AUC-PR **{mc_row.auc_pr:.4f}**, ECE
**{mc_row.ece:.4f}**, NLL **{mc_row.nll:.4f}**.

Gate M remains closed because the frozen packed training/validation dataset and upstream SCI v2
derivative are absent. TGN-style, TGAT-style, and FraudSAGE cannot be trained on the identical split;
validation-only temperature scaling cannot be fitted; the temperature-inclusive reliability figure
and all required paired comparisons are consequently incomplete. No test-set calibration fitting or
cross-dataset baseline substitution was used.
""",
    )
    write_text(
        MAIN_REPORTS / "data_recovery_requirement.md",
        f"""# Frozen Data Recovery Requirement

Expected packed path: `{DATASET}`. The manifest has been recovered, but `graph.pt`,
`transactions.parquet`, and `future_edge_audit.csv` are absent. Expected upstream derivative:
`{UPSTREAM}`; it is also absent.
The preserved manifest binds the packed graph to SHA-256 `{manifest['graph_sha256']}` and the
transaction metadata to `{manifest['transactions_sha256']}`, and the future-edge audit to
`{manifest['future_edge_audit_sha256']}`. Publication training must resume only after the
recovered/rebuilt files match these frozen hashes, or after a new explicitly versioned data freeze is
approved before looking at new test results. The missing audit must not be synthetically reconstructed
from its expected hash alone.
""",
    )
    return gate


def main() -> int:
    gate = run()
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
