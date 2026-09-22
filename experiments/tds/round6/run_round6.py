"""Build the final Round 6 evidence package and evaluate Gate M v7."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.round6.evidence import (
    build_model_metrics,
    build_positive_count_sensitivity,
    build_temporal_slice_metrics,
    package_preserved_artifacts,
)
from experiments.round6.latency import audit_latency
from experiments.round6.policy import evaluate_gate_v7
from experiments.round6.recovery import audit_recovery


DATASET = ROOT / "data" / "benchmark" / "gog_scimain_v1"
UPSTREAM = Path("/mnt/d/_Work/_data/GoG_sci_v2")
ROUND4 = ROOT / "results" / "graphrag" / "round_4"
RESULTS = ROOT / "results" / "main_final"
REPORTS = ROOT / "reports" / "main_final"
FIGURES = ROOT / "figures" / "main_final"
WORK_REPORT = ROOT / "docs" / "work_reports" / "309_graphRAG_dataset_round_6"
PAPER = ROOT / "docs" / "papers" / "_43_01_GraphRAG" / "graphrag.tex"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _complete(frame: pd.DataFrame, method: str) -> bool:
    selected = frame[frame.method == method]
    return bool(len(selected) == 5 and selected.status.str.startswith("COMPLETE").all())


def _paper_checks() -> tuple[bool, bool, str]:
    source = PAPER.read_text(encoding="utf-8")
    title = next(line for line in source.splitlines() if line.startswith("\\title{"))
    forbidden = ("Privacy-Preserving", "GraphRAG", "Cross-Layer", "Early-Warning")
    title_ok = not any(term.lower() in title.lower() for term in forbidden)
    format_ok = (
        "\\documentclass[10pt,journal]{IEEEtran}" in source
        and "\\bibliographystyle{IEEEtran}" in source
        and "\\bibliographystyle{elsarticle-num}" not in source
    )
    return title_ok, format_ok, title


def run(n_resamples: int = 10_000) -> dict[str, object]:
    for directory in (RESULTS, REPORTS, FIGURES, WORK_REPORT):
        directory.mkdir(parents=True, exist_ok=True)

    recovery = audit_recovery(
        DATASET, UPSTREAM, ROUND4 / "real_dataset_manifest.json",
        RESULTS / "dataset_recovery_manifest.json",
        REPORTS / "data_recovery_audit.md",
    )
    package = package_preserved_artifacts(ROUND4, RESULTS)
    model_metrics = build_model_metrics(RESULTS)
    temporal_slices = build_temporal_slice_metrics(RESULTS)
    sensitivity = build_positive_count_sensitivity(ROUND4, RESULTS, n_resamples)
    latency = audit_latency(
        ROUND4 / "mc_sensitivity.csv",
        ROOT / "results" / "graphrag" / "round_3" / "real_e2e_latency.csv",
        RESULTS / "latency_definition.json",
        FIGURES / "mc_tradeoff.png",
    )

    calibration = pd.read_csv(RESULTS / "calibration_baselines.csv")
    temporal = pd.read_csv(RESULTS / "temporal_baselines.csv")
    fraud = pd.read_csv(RESULTS / "fraud_specific_baseline.csv")
    comparisons = pd.read_csv(RESULTS / "statistical_comparisons.csv")
    gate_v6 = json.loads((ROOT / "results" / "paper_ready_gate_v6.json").read_text())
    title_ok, format_ok, title = _paper_checks()

    ordinary = comparisons[
        (comparisons.status == "COMPLETE")
        & (comparisons.bootstrap_scheme == "ordinary paired event bootstrap")
    ]
    available_panel_bootstrap = bool(
        len(ordinary) >= 2 and (ordinary.n_bootstrap == n_resamples).all()
    )
    available_panel_randomization = bool(
        len(ordinary) >= 2 and (ordinary.n_randomization == n_resamples).all()
    )
    required_names = {
        "Proposed deterministic vs TGN",
        "Proposed deterministic vs TGAT",
        "Proposed deterministic vs FraudSAGE",
        "MC Dropout T=10 vs Deterministic GNN",
        "MC Dropout T=10 vs Temperature Scaling",
        "MC Dropout Ensemble T=10 vs Deep Ensemble",
        "Deep Ensemble vs best temporal baseline",
    }
    completed_names = set(comparisons.loc[comparisons.status == "COMPLETE", "comparison"])
    required_comparisons = required_names.issubset(completed_names)

    checks = {
        "dataset_exact_recovery": bool(recovery["dataset_exact_recovery"]),
        "new_dataset_version_fully_retrained": bool(recovery["new_dataset_version_fully_retrained"]),
        "future_edge_audit_verified": bool(recovery["future_edge_audit_verified"]),
        "chronological_5seed_evaluation_complete": package["checkpoints"]["seed_count"] == 5,
        "tgn_complete": _complete(temporal, "TGN-style event memory"),
        "tgat_complete": _complete(temporal, "TGAT-style"),
        "fraud_specific_baseline_complete": bool(
            len(fraud) == 5 and fraud.status.str.startswith("COMPLETE").all()
        ),
        "temperature_scaling_complete": _complete(calibration, "Temperature-Scaled GNN"),
        "deep_ensemble_complete": bool(
            ((calibration.method == "Deep Ensemble") & calibration.status.str.startswith("COMPLETE")).any()
        ),
        "available_panel_bootstrap_complete": available_panel_bootstrap,
        "available_panel_randomization_complete": available_panel_randomization,
        "gate_required_model_comparisons_complete": required_comparisons,
        "paired_bootstrap_complete": required_comparisons,
        "randomization_analysis_complete": required_comparisons and available_panel_randomization,
        "positive_count_sensitivity_complete": len(sensitivity) == 2,
        "temporal_slice_available_panel_complete": bool(
            (temporal_slices.status == "COMPLETE").sum() == 12
        ),
        "reliability_figures_complete": bool(
            _complete(calibration, "Temperature-Scaled GNN")
            and (FIGURES / "reliability_comparison.png").is_file()
        ),
        "latency_scope_consistent": bool(latency["latency_scope_consistent"]),
        "main_evidence_package_complete_for_available_panel": all((
            (RESULTS / "model_metrics.csv").is_file(),
            (RESULTS / "temporal_slice_metrics.csv").is_file(),
            (RESULTS / "checkpoints_manifest.json").is_file(),
            (RESULTS / "raw_predictions" / "manifest.json").is_file(),
        )),
        "title_claims_match_evidence": title_ok,
        "publication_format_consistent": format_ok,
        "independent_benign_adjudication": False,
        "common_support_balance": False,
        "semantic_alignment_test": False,
        "same_corpus_retrieval_baseline": bool(gate_v6.get("same_corpus_retrieval_baseline", False)),
        "real_wallet_transactions": False,
        "real_hash_timestamp_lineage": False,
        "scamwallet_onchain_v1": False,
        "dlg_gnn_5seed": False,
        "dlg_permutation_sanity": False,
        "cross_layer_complete_cases": False,
    }
    gate = evaluate_gate_v7(checks)
    gate["data_recovery_decision"] = recovery["decision"]
    gate["paper_title"] = title
    gate["blocking_reasons"] = [
        key for key in (
            "dataset_exact_recovery", "future_edge_audit_verified", "tgn_complete",
            "tgat_complete", "fraud_specific_baseline_complete", "temperature_scaling_complete",
            "gate_required_model_comparisons_complete", "paired_bootstrap_complete",
            "randomization_analysis_complete", "reliability_figures_complete",
        ) if not gate.get(key, False)
    ]
    (ROOT / "results" / "paper_ready_gate_v7.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _write(REPORTS / "publication_readiness_report_v7.md", f"""# Main Paper Gate M v7

Gate M: **{gate['gate_m_main_timestamp_gnn']}**. Exact v1 recovery:
**{recovery['dataset_exact_recovery']}**. The hash-first search found no exact graph, transaction,
split, future-edge-audit, or upstream derivative artifact. No v1.1 numbers were generated because a
provenance-complete upstream source is also unavailable.

The preserved five-checkpoint test panel remains valid and is now packaged with raw prediction and
checkpoint hashes. Ordinary and class-stratified 10,000-resample bootstrap analyses are complete for
the two available comparisons. Gate-required temporal/fraud/calibration comparisons remain blocked;
the completed-panel statistics and full-gate statistics are represented by separate fields.

The MC runtime figure is now explicitly labeled as full 3,648-event held-out-panel elapsed time. It
is not presented as single-event end-to-end latency. Full manuscript result finalization remains
prohibited until Gate M v7 passes.
""")

    _write(WORK_REPORT / "implementation_plan.md", """# Round 6 Implementation Plan

1. Search exact paths, backups, archives, Git objects, and hash candidates before rebuilding.
2. Verify recovered artifacts against the preserved v1 SHA-256 contract.
3. Apply Option B3 if neither exact v1 nor a provenance-complete upstream derivative exists.
4. Package all still-valid five-checkpoint held-out evidence without refitting.
5. Add class-stratified positive-count sensitivity and explicit temporal blocked rows.
6. Audit and relabel the MC figure as full-panel batched runtime.
7. Evaluate Gate M/A/B independently in Gate v7; finalize the manuscript only if Gate M passes.
""")
    _write(WORK_REPORT / "task.md", f"""# Round 6 Task Status

- [x] Search exact frozen paths and local backup/archive candidates.
- [x] Perform extension-constrained SHA-256 candidate search.
- [x] Produce the read-only data recovery audit and manifest.
- [ ] Recover exact graph, transaction, split, and future-edge audit artifacts.
- [ ] Train TGN-style and TGAT-style baselines on the frozen protocol.
- [ ] Train the fraud-oriented GraphSAGE baseline on the frozen protocol.
- [ ] Fit per-seed validation-only temperature scaling.
- [x] Package preserved checkpoints and T=1/T=10 raw predictions with hashes.
- [x] Complete ordinary and class-stratified bootstrap for the available panel.
- [x] Separate available-panel statistics from Gate-required comparisons.
- [x] Audit and relabel the MC runtime measurement scope.
- [x] Generate Gate M v7 independently from Scam Gates A/B.
- [ ] Finalize manuscript results (blocked while Gate M is false).
""")
    _write(WORK_REPORT / "walkthrough.md", f"""# Round 6 Walkthrough

## Outcome

Gate M v7 is **{gate['gate_m_main_timestamp_gnn']}**. Hash-first recovery exhausted the recorded
project locations, local work archives, common Windows user backup folders, Git history/unreachable
objects, the D: recycle bin, and an extension-constrained SHA-256 search. Only byte-identical copies
of `real_dataset_manifest.json` were found. The required `graph.pt`, `transactions.parquet`,
`split_manifest.json`, `future_edge_audit.csv`, and GoG SCI v2 derivative remain unavailable.

Option B3 therefore applies. The v1 benchmark was not overwritten, a v1.1 benchmark was not claimed,
and no temporal, fraud-specific, or temperature-scaling result was manufactured.

## Evidence completed without retraining

The five-checkpoint, 3,648-event/107-positive held-out panel is packaged under `results/main_final`
with file hashes. The existing ordinary 10,000-resample comparisons were preserved and corresponding
class-stratified bootstrap sensitivity was added. Six temporal bins explicitly contain complete Deep
Ensemble/MC rows and blocked temporal/temperature rows.

The previous MC timing is the elapsed model-inference loop for the complete 3,648-event panel, batched
at 128 events. It differs from the legacy controlled single-event end-to-end timing. The figure and
metadata now name this scope explicitly, so the two quantities are not mixed.

## Publication rule

Gate A and Gate B remain false by design. Gate M remains false because the frozen evidence and all
training-dependent comparisons are incomplete. The manuscript title/format cleanup remains valid,
but final Results/Discussion revision was not performed.
""")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    args = parser.parse_args()
    gate = run(args.n_resamples)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
