"""Export the exact cascade artifact names required by the R2 closure task."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from validation.sci_v3_final_common import atomic_csv, atomic_json


ROOT = Path("results/sci_v3_submission_r2")
CASCADE = ROOT / "cascade"


def main() -> None:
    metrics = pd.read_csv(CASCADE / "calibrated_cascade_metrics.csv")
    trace = pd.read_csv(CASCADE / "interface_trace.csv")
    distributions = pd.read_csv(CASCADE / "score_distributions.csv")

    augmented = trace.copy()
    augmented["fast_path"] = augmented["fast_model"]
    augmented["raw_output_type"] = "probability"
    augmented["raw_output_range"] = "[0,1] (verified by score-semantics tests)"
    augmented["calibrated_output_type"] = "probability"
    augmented["fusion_input_type"] = "calibrated_log_odds"
    augmented["router_input_type"] = "calibrated_fast_probability"
    augmented["decision_threshold"] = augmented["final_threshold"]
    augmented["tau_b"] = pd.NA
    augmented["tau_f"] = pd.NA
    augmented["tau_u"] = augmented["route_cutoff"]
    augmented["router_rule"] = "abs(p_fast-fast_threshold)<=tau_u"
    augmented["deep_representation_source"] = "ProductionLevel1GIN meanmax embedding + GIN score"
    augmented["deep_representation_dim"] = 65
    augmented["fusion_lambda"] = augmented["fast_weight"]
    augmented["n_validation"] = 3648
    augmented["n_test"] = 3648
    ordered = [
        "seed", "fast_path", "raw_output_type", "raw_output_range",
        "calibrated_output_type", "fusion_input_type", "router_input_type",
        "decision_threshold", "tau_b", "tau_f", "tau_u", "router_rule",
        "deep_representation_source", "deep_representation_dim", "fusion_lambda",
        "n_validation", "n_test",
    ] + [column for column in augmented.columns if column not in {
        "fast_model", *[
            "seed", "fast_path", "raw_output_type", "raw_output_range",
            "calibrated_output_type", "fusion_input_type", "router_input_type",
            "decision_threshold", "tau_b", "tau_f", "tau_u", "router_rule",
            "deep_representation_source", "deep_representation_dim", "fusion_lambda",
            "n_validation", "n_test",
        ]
    }]
    atomic_csv(CASCADE / "interface_trace.csv", augmented[ordered])
    atomic_csv(CASCADE / "score_distribution_summary.csv", distributions)
    atomic_csv(CASCADE / "cascade_per_seed.csv", metrics)

    numeric = [
        "roc_auc", "pr_auc", "f1", "precision", "fraud_recall", "fnr", "mcc",
        "balanced_accuracy", "deep_route_rate", "direct_exit_rate",
    ]
    summary_rows: list[dict[str, object]] = []
    for (model, policy), group in metrics.groupby(["model", "routing_policy"], sort=False):
        row: dict[str, object] = {
            "model": model,
            "routing_policy": policy,
            "n_seeds": int(group.seed.nunique()),
            "n_test_per_seed": int(group.N.iloc[0]),
            "n_fraud_per_seed": int(group.N_positive.iloc[0]),
        }
        for column in numeric:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_sd"] = float(group[column].std(ddof=1))
        summary_rows.append(row)
    atomic_csv(CASCADE / "cascade_summary.csv", pd.DataFrame(summary_rows))

    calibration = {
        "method": "Platt calibration on validation log-odds",
        "selection_partition": "validation_only",
        "test_labels_used_for_selection": False,
        "fusion_space": "calibrated_log_odds",
        "interface_case": "A",
        "deep_representation": {
            "source": "ProductionLevel1GIN meanmax embedding concatenated with GIN score",
            "dimension": 65,
        },
        "router_rule": "abs(p_fast-fast_threshold)<=route_cutoff",
        "tau_semantics": {
            "tau_b": "not used by this ambiguity-budget router",
            "tau_f": "not used by this ambiguity-budget router",
            "tau_u": "route_cutoff",
        },
        "per_seed": trace.to_dict(orient="records"),
    }
    atomic_json(CASCADE / "cascade_calibration.json", calibration)

    cascade_rows = metrics[metrics.routing_policy == "validation_calibrated"].copy()
    cascade_rows["N_direct"] = (cascade_rows.N * cascade_rows.direct_exit_rate).round().astype(int)
    cascade_rows["N_deep"] = (cascade_rows.N * cascade_rows.deep_route_rate).round().astype(int)
    fraud_direct: dict[int, int] = {}
    for seed in cascade_rows.seed.astype(int):
        predictions = pd.read_csv(CASCADE / f"predictions/ProductionLevel1GIN__seed{seed}.csv")
        fraud_direct[seed] = int(((predictions.label == 1) & (predictions.deep_executed == 0)).sum())
    cascade_rows["N_fraud_direct"] = cascade_rows.seed.astype(int).map(fraud_direct)

    timing = pd.read_csv(ROOT / "runtime/five_repeat_policy_timing.csv")
    timing = timing[timing.policy == "validation_calibrated_dual"]
    timing_summary = timing.groupby("seed", as_index=False).agg(
        runtime_deep_route_rate=("deep_route_rate", "mean"),
        measured_e2e_mean_ms=("mean_latency_ms", "mean"),
        measured_e2e_p95_ms=("p95_latency_ms", "mean"),
        measured_e2e_p99_ms=("p99_latency_ms", "mean"),
        throughput_events_s=("throughput_events_per_second", "mean"),
        rss_peak_bytes=("rss_peak_bytes", "max"),
        vram_peak_bytes=("vram_peak_bytes", "max"),
    )
    frontier = cascade_rows.merge(timing_summary, on="seed", how="left")
    frontier = frontier.rename(columns={"deep_route_rate": "test_deep_route_rate"})
    frontier["classification_population"] = "held_out_graph_test"
    frontier["runtime_population"] = "fixed_500_event_all_benign_prefix"
    frontier["runtime_positive_support"] = 0
    frontier["runtime_accuracy_status"] = "UNDEFINED_SINGLE_CLASS_TARGET"
    frontier["runtime_measurement_type"] = "warm_started_measured_e2e_5_repeats"
    atomic_csv(CASCADE / "cascade_frontier.csv", frontier)

    print(json.dumps({
        "interface_rows": len(augmented),
        "per_seed_rows": len(metrics),
        "summary_rows": len(summary_rows),
        "frontier_rows": len(frontier),
    }, indent=2))


if __name__ == "__main__":
    main()
