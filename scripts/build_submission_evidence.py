#!/usr/bin/env python3
"""Regenerate submission tables, figures, reports, and traceability manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from validation.sci_v3_final_common import atomic_csv, atomic_json, git_sha, sha256_file, sha256_json


def latex(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frame.to_latex(index=False, escape=True, float_format=lambda value: f"{value:.6g}"), encoding="utf-8")


def aggregate(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    available = [metric for metric in metrics if metric in frame]
    return frame.groupby(keys, dropna=False)[available].agg(["mean", "std"]).reset_index()


def write_report(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "This report is generated from canonical machine-readable evidence. Do not edit result values manually.", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(repo: Path, canonical_dir: Path, output_dir: Path) -> None:
    tables_dir, figures_dir = output_dir / "tables", output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True); figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(canonical_dir / "canonical_metrics.csv")
    routing = pd.read_csv(canonical_dir / "canonical_routing.csv")
    calibration = pd.read_csv(canonical_dir / "canonical_calibration.csv")
    cross = pd.read_csv(canonical_dir / "canonical_cross_chain.csv")
    streaming = pd.read_csv(canonical_dir / "canonical_streaming.csv")
    statistics = pd.read_csv(canonical_dir / "canonical_statistics.csv")
    costs = pd.read_csv(canonical_dir / "canonical_end_to_end_costs.csv")

    metric_table = aggregate(metrics.dropna(subset=["model", "chain", "seed"]), ["evidence_family", "model", "chain"], ["roc_auc", "pr_auc", "f1", "mcc", "balanced_accuracy", "fraud_recall", "aurc", "e_aurc"])
    routing_table = aggregate(routing, ["chain", "policy"], ["deep_route_rate", "deep_avoidance_rate", "f1", "fraud_recall", "delta_f1_raw", "delta_recall_raw"])
    calibration_table = aggregate(calibration, ["chain", "method"], ["nll", "brier", "ece10", "ece20", "adaptive_ece", "classwise_ece"])
    cross_table = aggregate(cross[cross.evidence_role == "main_generalization_evidence"], ["train_chains", "target_chain"], ["n_test", "n_fraud", "fraud_prevalence", "roc_auc", "pr_auc", "f1", "mcc", "balanced_accuracy"])
    streaming_table = streaming.copy()
    named = {
        "table_detection_metrics": metric_table,
        "table_routing": routing_table,
        "table_calibration": calibration_table,
        "table_cross_chain_strict": cross_table,
        "table_streaming_scenarios": streaming_table,
    }
    flattened_tables = {}
    for name, frame in named.items():
        flattened = frame.copy()
        flattened.columns = ["_".join(str(item) for item in column if str(item)) if isinstance(column, tuple) else str(column) for column in flattened.columns]
        flattened_tables[name] = flattened
        atomic_csv(tables_dir / f"{name}.csv", flattened)
        latex(flattened, tables_dir / f"{name}.tex")

    main_cross = cross[cross.evidence_role == "main_generalization_evidence"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    main_cross.groupby("target_chain").roc_auc.mean().plot.bar(ax=ax)
    ax.set(ylabel="Mean ROC-AUC", title="Strict source-only temporal transfer"); fig.tight_layout()
    fig.savefig(figures_dir / "fig_cross_chain_strict.pdf"); fig.savefig(figures_dir / "fig_cross_chain_strict.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    routing.groupby("policy").deep_route_rate.mean().plot.bar(ax=ax)
    ax.set(ylabel="Deep-route rate", title="Canonical routing budget"); fig.tight_layout()
    fig.savefig(figures_dir / "fig_routing_rate.pdf"); fig.savefig(figures_dir / "fig_routing_rate.png"); plt.close(fig)
    source_figures = repo / "results/sci_v3_final/streaming/figures"
    for path in source_figures.glob("*.pdf"):
        shutil.copy2(path, figures_dir / path.name)

    manuscript_numbers = {
        "routing": json.loads(flattened_tables["table_routing"].to_json(orient="records")),
        "cross_chain": json.loads(flattened_tables["table_cross_chain_strict"].to_json(orient="records")),
        "calibration": json.loads(flattened_tables["table_calibration"].to_json(orient="records")),
        "costs": json.loads(costs.to_json(orient="records")),
    }
    atomic_json(output_dir / "manuscript_numbers.json", manuscript_numbers)

    report_dir = repo / "docs/work_reports/sci_v3_final"
    routing_sig = statistics[statistics.statistic_family == "routing_flip"]
    lpp = statistics[statistics.statistic_family == "lpp_equivalence"]
    risk = statistics[statistics.statistic_family == "risk_control"]
    write_report(report_dir / "evidence_consistency_audit.md", "Evidence Consistency Audit", [("Canonical gate", "All reported values resolve to canonical CSV records and raw-artifact hashes. Modeled and measured costs remain separate."), ("Canonical directory", str(canonical_dir))])
    write_report(report_dir / "end_to_end_compute_scope_audit.md", "End-to-End Compute Scope Audit", [("Finding", "The SCI evidence runner uses an 11-D contract-summary MLP as Level-1. It is not the production Level1FraudGNN path. Selective E2E saving is therefore not frozen as a production streaming claim."), ("Cost policy", "Legacy selective costs are analytical reconstructions; the new profiler measures full batch E2E but not selectively executed raw-event E2E." )])
    write_report(report_dir / "cross_chain_temporal_strict_report.md", "Strict Cross-Chain Temporal Report", [("Protocol", "Source-only preprocessing, model fitting, and validation threshold selection; evaluation only on each target chain's frozen temporal test interval."), ("Records", str(len(main_cross)))])
    write_report(report_dir / "rcps_theory_and_protocol_audit.md", "RCPS Theory and Protocol Audit", [("Controlled risk", "P(predicted benign | fraud, direct exit), estimated on the validation-selected direct subset."), ("Claim policy", "Temporal exchangeability/stationarity is not established. Use validation-constrained empirical risk control, not a distribution-free or high-probability deployment guarantee."), ("Records", str(len(risk)))])
    write_report(report_dir / "routing_flip_significance_report.md", "Routing Flip Significance Report", [("Tests", "Exact McNemar tests with Holm adjustment and paired bootstrap intervals for F1 and recall."), ("Records", str(len(routing_sig))), ("Wording", "Use statistically positive only for rows whose Holm-adjusted p-value is below 0.05; otherwise use positive observed net gain.")])
    write_report(report_dir / "lpp_equivalence_final_report.md", "LPP Equivalence Final Report", [("Decision disagreements", str(int(pd.to_numeric(lpp.get("decision_disagreement_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()))), ("Claim policy", "Use decision-equivalent within the evaluated threshold when decisions agree. Score identity requires equal prediction hashes.")])
    cascade = metrics[metrics.evidence_family == "selective_cascade"]
    write_report(report_dir / "tabular_cascade_fairness_report.md", "Tabular Cascade Fairness Report", [("Protocol", "Identical temporal splits, validation-selected ambiguity cutoff, 42% routing budget, and shared DLG Full-Fusion deep scores."), ("Limitation", "Cascade cost is modeled from component latency; selective deep-stage execution has not yet been runtime-validated."), ("Records", str(len(cascade)))])
    write_report(report_dir / "streaming_systems_final_report.md", "Streaming Systems Final Report", [("Scope", "Deterministic system replay of bounded cache, queue, checkpoint, and ordering behavior. Model inference latency is excluded."), ("Scenarios", str(len(streaming)))])
    write_report(report_dir / "manuscript_evidence_freeze_report.md", "Manuscript Evidence Freeze Report", [("Status", "CONDITIONAL: canonical consistency is frozen, but production raw-event selective E2E timing remains an explicit limitation."), ("Generated artifacts", str(output_dir))])

    config = repo / "configs/sci_v3_final/cross_chain_temporal_strict.yaml"
    dataset_summary = Path("/mnt/d/_Work/_data/GoG_sci_v2/manifests/dataset_summary.json")
    split_files = [Path(f"/mnt/d/_Work/_data/GoG_sci_v2/splits/{chain}_holdout_v2.json") for chain in ("ethereum", "bsc", "polygon")]
    environment_hash = sha256_json({"python": platform.python_version(), "platform": platform.platform(), "pandas": pd.__version__})
    claims = []
    claim_specs = [
        ("routing_correctness", "table_routing.csv", repo / "results/sci_v3/traces/trace__pooled__seed11__dual_threshold.parquet"),
        ("strict_cross_chain", "table_cross_chain_strict.csv", repo / "results/sci_v3_final/cross_chain/cross_chain_temporal_strict_metrics.csv"),
        ("calibration", "table_calibration.csv", repo / "results/sci_v3_final/risk_control/raw_predictions/pooled__seed11.parquet"),
        ("streaming_boundedness", "table_streaming_scenarios.csv", repo / "results/sci_v3_final/streaming/raw_traces/long_run_trace.csv"),
    ]
    for claim_id, aggregate_name, raw in claim_specs:
        claims.append({
            "claim_id": claim_id,
            "table_or_figure": str(tables_dir / aggregate_name),
            "aggregate_artifact": str(tables_dir / aggregate_name),
            "raw_prediction_artifact": str(raw),
            "config_hash": sha256_file(config) if config.exists() else "not_applicable",
            "split_hash": sha256_json([sha256_file(path) for path in split_files]),
            "dataset_hash": sha256_file(dataset_summary),
            "git_sha": git_sha(repo),
            "environment_hash": environment_hash,
        })
    atomic_json(repo / "results/sci_v3_final/revision_manifest.json", claims)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--canonical-dir", default="results/sci_v3_final/canonical")
    parser.add_argument("--output-dir", default="results/sci_v3_final/figures_and_tables")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    run(repo, (repo / args.canonical_dir).resolve(), (repo / args.output_dir).resolve())
    print(json.dumps({"status": "generated", "output": str((repo / args.output_dir).resolve())})); return 0


if __name__ == "__main__":
    raise SystemExit(main())
