"""Build manuscript-ready SCI-v3 R2 tables, figures, and claim manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from validation.sci_v3_final_common import atomic_csv, atomic_json


METRICS = ["roc_auc", "pr_auc", "f1", "fraud_recall", "precision", "mcc", "balanced_accuracy"]


def summarize(frame: pd.DataFrame, keys: list[str], metrics: list[str] = METRICS) -> pd.DataFrame:
    available = [metric for metric in metrics if metric in frame.columns]
    result = frame.groupby(keys, dropna=False)[available].agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(str(item) for item in column if item).rstrip("_") if isinstance(column, tuple) else column
                      for column in result.columns]
    return result


def save_table(path: Path, frame: pd.DataFrame) -> None:
    atomic_csv(path.with_suffix(".csv"), frame)
    path.with_suffix(".tex").parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".tex").write_text(frame.to_latex(index=False, float_format=lambda value: f"{value:.3f}", escape=True), encoding="utf-8")


def reliability(predictions: pd.DataFrame, score_name: str) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0, 1, 11)
    confidence, frequency = [], []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (predictions[score_name] >= lower) & (predictions[score_name] < upper if upper < 1 else predictions[score_name] <= upper)
        if mask.any():
            confidence.append(predictions.loc[mask, score_name].mean())
            frequency.append(predictions.loc[mask, "label"].mean())
    return np.asarray(confidence), np.asarray(frequency)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sci_v3_submission_r2/closure.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(cfg["output_root"])
    table_root, figure_root = root / "manuscript/tables", root / "manuscript/figures"
    table_root.mkdir(parents=True, exist_ok=True); figure_root.mkdir(parents=True, exist_ok=True)
    canonical_root = Path("results/sci_v3_final/canonical")
    canonical = pd.read_csv(canonical_root / "canonical_metrics.csv")
    routing = pd.read_csv(canonical_root / "canonical_routing.csv")
    cross = pd.read_csv(canonical_root / "canonical_cross_chain.csv")
    calibration = pd.read_csv(canonical_root / "canonical_calibration.csv")
    statistics = pd.read_csv(canonical_root / "canonical_statistics.csv")
    runtime = pd.read_csv("results/sci_v3_submission/canonical/canonical_runtime.csv")
    streaming = pd.read_csv("results/sci_v3_submission/canonical/canonical_streaming.csv")
    new_metrics = pd.read_csv(root / "cascade/calibrated_cascade_metrics.csv")
    new_calibration = pd.read_csv(root / "cascade/calibration_metrics.csv")
    seed_pairs = pd.read_csv(root / "statistics/seed_pairs.csv")
    mcnemar = pd.read_csv(root / "statistics/mcnemar_holm.csv")
    claim_status = json.loads((root / "statistics/claim_status.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "cascade/acceptance_gate.json").read_text(encoding="utf-8"))

    supervised = canonical[canonical.evidence_family == "supervised_gnn_baseline"]
    tabular = canonical[canonical.evidence_family == "tabular_baseline"]
    gad = canonical[canonical.evidence_family == "legacy_main_raw_predictions"]
    save_table(table_root / "table_supervised_baselines", summarize(supervised, ["chain", "model"]))
    save_table(table_root / "table_tabular_controls", summarize(tabular, ["chain", "model"]))
    save_table(table_root / "table_graph_anomaly_baselines", summarize(gad, ["chain", "model"]))
    save_table(table_root / "table_routing_summary", summarize(routing, ["chain", "policy", "T"],
        ["deep_route_rate", "wrong_to_correct", "correct_to_wrong", "roc_auc", "pr_auc", "f1", "fraud_recall"]))
    flip = statistics[statistics.statistic_family == "routing_flip"].copy()
    save_table(table_root / "table_routing_flip_statistics", flip[["chain", "seed", "policy", "wrong_to_correct",
        "correct_to_wrong", "mcnemar_exact_p_value", "holm_adjusted_p_value", "f1_delta", "f1_delta_ci95_low",
        "f1_delta_ci95_high", "significance_established_0_05"]])
    save_table(table_root / "table_calibrated_production_cascade", summarize(new_metrics, ["model", "routing_policy"],
        ["deep_route_rate", "roc_auc", "pr_auc", "f1", "fraud_recall", "precision", "mcc"]))
    save_table(table_root / "table_calibrated_seed_pairs", seed_pairs)
    save_table(table_root / "table_calibrated_mcnemar_holm", mcnemar)
    save_table(table_root / "table_calibration", summarize(new_calibration, ["model", "split", "calibration"],
        ["brier", "log_loss", "ece_10", "roc_auc"]))

    strict = cross[cross.protocol == "strict_target_temporal_holdout"]
    save_table(table_root / "table_cross_chain", summarize(strict, ["train_chains", "target_chain"],
        ["roc_auc", "pr_auc", "f1", "fraud_recall", "precision", "mcc"]))
    save_table(table_root / "table_streaming_100k", streaming)
    save_table(table_root / "table_runtime_measured", summarize(runtime, ["scenario", "policy", "mc_samples"],
        ["mean_latency_ms", "p95_latency_ms", "p99_latency_ms", "throughput_events_per_second", "deep_route_rate",
         "rss_peak_bytes", "vram_peak_bytes"]))

    lpp_score = pd.read_csv("results/sci_v3_final/statistics/lpp_equivalence.csv")
    lpp_resource = pd.read_csv("results/results_sci_v2/streaming/streaming_resource_metrics.csv")
    lpp_resource = lpp_resource[(lpp_resource.scenario == "100k_event_long_run") & lpp_resource.variant.isin(["no_purge", "full_lpp_ttl_lru"])]
    lpp_summary = summarize(lpp_score, ["chain"], ["max_abs_score_diff", "mean_abs_score_diff", "p99_abs_score_diff",
        "decision_disagreement_rate", "ranking_order_change_count"])
    lpp_summary["all_prediction_hash_equal"] = lpp_score.groupby("chain").prediction_hash_equal.all().values
    save_table(table_root / "table_lpp_score_equivalence", lpp_summary)
    save_table(table_root / "table_lpp_resource", lpp_resource)

    # Dataset evidence available without reconstructing the deleted source root.
    dataset = canonical[canonical.chain.isin(["ethereum", "bsc", "polygon"])].dropna(subset=["n_test"])
    dataset = dataset.sort_values(["chain", "seed"]).groupby("chain").first().reset_index()
    dataset_table = dataset[["chain", "n_test", "n_fraud", "n_benign", "fraud_prevalence"]]
    dataset_table["scope"] = "frozen temporal test split"
    save_table(table_root / "table_dataset_test_audit", dataset_table)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(6.8, 3.8))
    axis.bar(dataset_table.chain, dataset_table.n_test, label="benign + fraud", color="#4C78A8")
    axis.bar(dataset_table.chain, dataset_table.n_fraud, label="fraud", color="#E45756")
    axis.set_ylabel("Test events"); axis.legend(); fig.tight_layout()
    fig.savefig(figure_root / "figure_dataset_test_counts.pdf", bbox_inches="tight"); fig.savefig(figure_root / "figure_dataset_test_counts.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    predictions = pd.concat([pd.read_csv(root / f"cascade/predictions/ProductionLevel1GIN__seed{seed}.csv").assign(seed=seed)
                             for seed in cfg["seeds"]], ignore_index=True)
    fig, axis = plt.subplots(figsize=(4.8, 4.3))
    for score, label, marker in (("raw_fast_score", "Raw GIN", "o"), ("calibrated_fast_score", "Calibrated GIN", "s"),
                                 ("final_score", "Selective fusion", "^")):
        confidence, frequency = reliability(predictions, score)
        axis.plot(confidence, frequency, marker=marker, label=label)
    axis.plot([0,1], [0,1], "--", color="black", linewidth=1); axis.set(xlabel="Mean predicted probability", ylabel="Observed fraud frequency")
    axis.legend(); fig.tight_layout(); fig.savefig(figure_root / "figure_reliability.pdf", bbox_inches="tight"); fig.savefig(figure_root / "figure_reliability.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.8, 4.2))
    for label, group in new_metrics.groupby("model"):
        axis.scatter(group.deep_route_rate, group.f1, label=label, alpha=.8)
    axis.set(xlabel="Deep-route rate", ylabel="F1", xlim=(-.02, 1.02)); axis.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figure_root / "figure_accuracy_cost_frontier.pdf", bbox_inches="tight"); fig.savefig(figure_root / "figure_accuracy_cost_frontier.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    risk = statistics[statistics.statistic_family == "risk_control"].dropna(subset=["coverage", "observed_test_risk"])
    fig, axis = plt.subplots(figsize=(6.3, 4.0))
    for policy, group in risk.groupby("policy"):
        ordered = group.sort_values("coverage"); axis.plot(ordered.coverage, ordered.observed_test_risk, marker="o", label=policy)
    axis.set(xlabel="Direct-exit coverage", ylabel="Observed direct-exit risk"); axis.legend(); fig.tight_layout()
    fig.savefig(figure_root / "figure_risk_coverage.pdf", bbox_inches="tight"); fig.savefig(figure_root / "figure_risk_coverage.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    trace_path = Path("results/sci_v3_submission/integrated/raw_trace_100k.csv")
    trace = pd.read_csv(trace_path, usecols=["rss_bytes", "store_bytes", "cache_bytes"])
    stride = max(1, len(trace)//2000); sampled = trace.iloc[::stride].copy(); sampled["event"] = sampled.index + 1
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.plot(sampled.event, sampled.rss_bytes/2**20, label="Process RSS")
    axis.plot(sampled.event, sampled.store_bytes/2**20, label="Bounded graph store")
    axis.plot(sampled.event, sampled.cache_bytes/2**20, label="Embedding cache")
    axis.set(xlabel="Processed event", ylabel="Memory (MiB)"); axis.legend(); fig.tight_layout()
    fig.savefig(figure_root / "figure_streaming_memory.pdf", bbox_inches="tight"); fig.savefig(figure_root / "figure_streaming_memory.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    claims = [
        {"claim_id": "C-PRODUCTION-BACKBONE", "status": "SUPPORTED", "claim": "production path uses bounded GIN to GATv2",
         "evidence": "cascade/interface_trace.csv"},
        {"claim_id": "C-GIN-CASCADE-F1", "status": claim_status["status"], "claim": "calibrated selective fusion improves GIN F1",
         "evidence": "statistics/claim_status.json"},
        {"claim_id": "C-TABULAR-CASCADE", "status": gate["gate"], "claim": "tabular triage cascades improve accuracy-cost trade-off",
         "evidence": "cascade/acceptance_gate.json"},
        {"claim_id": "C-LPP-EQUIVALENCE", "status": "SUPPORTED_LIMITED", "claim": "LPP is decision-equivalent at evaluated thresholds, not bitwise-identical",
         "evidence": "manuscript/tables/table_lpp_score_equivalence.csv"},
        {"claim_id": "C-STREAMING-BOUNDED", "status": "SUPPORTED", "claim": "100k-event run is loss-free and bounded under the measured configuration",
         "evidence": "manuscript/tables/table_streaming_100k.csv"},
        {"claim_id": "C-CROSS-CHAIN", "status": "DESCRIPTIVE", "claim": "strict source-only cross-chain transfer is reported without universal generalization claim",
         "evidence": "manuscript/tables/table_cross_chain.csv"},
    ]
    atomic_csv(root / "claim_manifest_v2.csv", pd.DataFrame(claims)); atomic_json(root / "claim_manifest_v2.json", {"version": 2, "claims": claims})
    atomic_json(root / "manuscript/artifact_manifest.json", {"tables": sorted(path.name for path in table_root.glob("*.csv")),
        "figures": sorted(path.name for path in figure_root.glob("*.pdf")), "tabular_cascade_gate": gate["gate"]})
    print(json.dumps({"tables": len(list(table_root.glob('*.csv'))), "figures": len(list(figure_root.glob('*.pdf'))), "gate": gate["gate"]}, indent=2))


if __name__ == "__main__":
    main()
