"""Package exact-name R2 deliverables without conflating evidence populations."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validation.sci_v3_final_common import atomic_csv, atomic_json, sha256_file


ROOT = Path("results/sci_v3_submission_r2")
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
PROFILING = ROOT / "profiling"
MANUSCRIPT_TABLES = ROOT / "manuscript/tables"
MANUSCRIPT_FIGURES = ROOT / "manuscript/figures"


def save_table(name: str, frame: pd.DataFrame) -> None:
    destination = TABLES / name
    atomic_csv(destination.with_suffix(".csv"), frame)
    destination.with_suffix(".tex").parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".tex").write_text(
        frame.to_latex(index=False, float_format=lambda value: f"{value:.3f}", escape=True),
        encoding="utf-8",
    )


def copy_table(source: str, destination: str) -> None:
    save_table(destination, pd.read_csv(MANUSCRIPT_TABLES / f"{source}.csv"))


def calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (scores >= lower) & (scores < upper if upper < 1.0 else scores <= upper)
        if mask.any():
            total += float(mask.mean()) * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return total


def adaptive_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    groups = np.array_split(np.argsort(scores), bins)
    return sum(
        len(group) / len(scores) * abs(float(labels[group].mean()) - float(scores[group].mean()))
        for group in groups if len(group)
    )


def build_routing_policy() -> pd.DataFrame:
    routing = pd.read_csv("results/sci_v3_final/canonical/canonical_routing.csv")
    routing = routing[(routing.chain == "pooled") & routing.policy.isin(["no_routing", "dual_threshold", "risk_sensitive"])].copy()
    direct_fraud = []
    for row in routing.itertuples():
        trace = pd.read_parquet(row.raw_trace_artifact, columns=["label", "route"])
        direct = trace.route != "deep_inspection"
        direct_fraud.append(int(((trace.label == 1) & direct).sum()))
    routing["n_direct_fraud"] = direct_fraud
    routing["n_direct_benign"] = routing.n_direct - routing.n_direct_fraud
    grouped = routing.groupby(["policy", "T"], as_index=False).agg(
        n_seeds=("seed", "nunique"), N_total=("n_total", "mean"), N_direct=("n_direct", "mean"),
        N_deep=("n_deep", "mean"), N_direct_benign=("n_direct_benign", "mean"),
        N_direct_fraud=("n_direct_fraud", "mean"), N_fraud_total=("n_fraud", "mean"),
        Deep_route_rate=("deep_route_rate", "mean"), Fraud_Recall=("fraud_recall", "mean"),
        FNR=("fnr", "mean"), F1=("f1", "mean"), PR_AUC=("pr_auc", "mean"), MCC=("mcc", "mean"),
    )
    grouped["Coverage"] = 1.0 - grouped.Deep_route_rate
    grouped["Selective_risk"] = pd.NA
    grouped["Risk_upper_bound"] = pd.NA
    grouped["alpha"] = pd.NA
    grouped["delta"] = pd.NA
    risk = pd.read_csv("results/sci_v3_final/canonical/canonical_statistics.csv")
    risk = risk[(risk.statistic_family == "risk_control") & (risk.chain == "pooled") & (risk.target_alpha == 0.05)]
    mask = grouped.policy == "risk_sensitive"
    if len(risk):
        grouped.loc[mask, "Selective_risk"] = float(risk.observed_test_risk.mean())
        grouped.loc[mask, "Risk_upper_bound"] = float(risk.test_upper_bound_diagnostic.mean())
        grouped.loc[mask, "alpha"] = 0.05
        grouped.loc[mask, "delta"] = float(risk.delta.mean())
    grouped["Undefined_reason"] = ""
    return grouped


def build_frontier(policy: pd.DataFrame) -> pd.DataFrame:
    timing = pd.read_csv(ROOT / "runtime/five_repeat_policy_summary.csv")
    mapping = {
        "no_routing": "full_deep",
        "dual_threshold": "legacy_dual",
        "risk_sensitive": "legacy_risk_controlled",
    }
    rows = []
    for classification in policy.itertuples(index=False):
        runtime = timing[timing.policy == mapping[classification.policy]].iloc[0]
        rows.append({
            "Policy": classification.policy,
            "classification_MC_T": int(classification.T),
            "runtime_MC_T": 1,
            "N_total_classification": classification.N_total,
            "N_direct_classification": classification.N_direct,
            "N_deep_classification": classification.N_deep,
            "Deep_route_rate_classification": classification.Deep_route_rate,
            "N_fraud_total": classification.N_fraud_total,
            "N_fraud_direct": classification.N_direct_fraud,
            "Fraud_Recall": classification.Fraud_Recall,
            "FNR": classification.FNR,
            "F1": classification.F1,
            "PR_AUC": classification.PR_AUC,
            "MCC": classification.MCC,
            "runtime_events_per_repeat": 500,
            "runtime_positive_support": 0,
            "runtime_deep_route_rate": runtime.deep_route_rate,
            "Mean_E2E_ms": runtime.mean_latency_ms,
            "P95_E2E_ms": runtime.p95_latency_ms,
            "P99_E2E_ms": runtime.p99_latency_ms,
            "Throughput_events_s": runtime.throughput_events_s,
            "Peak_RSS_bytes": runtime.rss_peak_bytes,
            "Peak_VRAM_bytes": runtime.vram_peak_bytes,
            "Latency_reduction_vs_full_deep_pct": runtime.latency_reduction_vs_full_deep_pct,
            "latency_provenance": "measured_e2e",
            "population_boundary": "classification=pooled held-out graphs; runtime=fixed all-benign raw prefix",
        })
    return pd.DataFrame(rows)


def build_calibration_table() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "cascade/predictions").glob("ProductionLevel1GIN__seed*.csv")):
        seed = int(path.stem.split("seed")[-1])
        data = pd.read_csv(path)
        labels = data.label.to_numpy(dtype=int)
        for score_name in ("raw_fast_score", "calibrated_fast_score", "final_score"):
            scores = data[score_name].to_numpy(dtype=float)
            rows.append({
                "seed": seed,
                "score": score_name,
                "N": len(labels),
                "N_positive": int(labels.sum()),
                "ECE_10": calibration_error(labels, scores, 10),
                "ECE_20": calibration_error(labels, scores, 20),
                "adaptive_ECE_10": adaptive_calibration_error(labels, scores),
                "classwise_ECE": np.mean([
                    abs(float(scores[labels == label].mean()) - float(label)) for label in (0, 1)
                ]),
            })
    return pd.DataFrame(rows)


def build_streaming_figure() -> None:
    trace = pd.read_csv(
        "results/sci_v3_submission/integrated/raw_trace_100k.csv",
        usecols=["total_latency_ms", "rss_bytes"],
    )
    stride = max(1, len(trace) // 2000)
    sampled = trace.iloc[::stride].copy()
    sampled["event"] = sampled.index + 1
    sampled["throughput_events_s"] = 1000.0 / trace.total_latency_ms.rolling(1000, min_periods=50).mean().iloc[::stride].to_numpy()
    slope, intercept = np.polyfit(sampled.event, sampled.rss_bytes, 1)
    figure, left = plt.subplots(figsize=(7.2, 4.2))
    right = left.twinx()
    left.plot(sampled.event, sampled.rss_bytes / 2**30, color="#4C78A8", label="RSS")
    left.plot(sampled.event, (intercept + slope * sampled.event) / 2**30, "--", color="#1F4E79", label="RSS fit")
    right.plot(sampled.event, sampled.throughput_events_s, color="#E45756", alpha=0.7, label="Rolling throughput")
    left.set(xlabel="Processed event", ylabel="Process RSS (GiB)")
    right.set_ylabel("Events/s (1000-event rolling mean)")
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / "fig_streaming_rss_throughput.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / "fig_streaming_rss_throughput.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_risk_figures() -> None:
    stats = pd.read_csv("results/sci_v3_final/canonical/canonical_statistics.csv")
    risk = stats[stats.statistic_family == "risk_control"].dropna(subset=["coverage", "observed_test_risk"]).copy()
    atomic_csv(ROOT / "risk_coverage_source.csv", risk)
    figure, axis = plt.subplots(figsize=(6.3, 4.0))
    for chain, group in risk.groupby("chain"):
        summary = group.groupby("coverage", as_index=False).observed_test_risk.mean().sort_values("coverage")
        axis.plot(summary.coverage, summary.observed_test_risk, marker="o", label=chain)
    axis.set(xlabel="Direct-exit coverage", ylabel="Observed direct-exit fraud-miss risk")
    axis.legend(fontsize=8); figure.tight_layout()
    figure.savefig(FIGURES / "fig_fnr_coverage.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / "fig_fnr_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_rss_accounting() -> pd.DataFrame:
    streaming = pd.read_csv(MANUSCRIPT_TABLES / "table_streaming_100k.csv").iloc[0]
    controlled = int(streaming.max_store_bytes) + int(streaming.max_cache_bytes)
    residual = int(streaming.rss_peak_bytes) - controlled
    frame = pd.DataFrame([
        {"component": "process_rss_peak", "bytes": int(streaming.rss_peak_bytes), "accounting": "measured_process_total"},
        {"component": "bounded_graph_store", "bytes": int(streaming.max_store_bytes), "accounting": "measured_logical_state"},
        {"component": "embedding_cache", "bytes": int(streaming.max_cache_bytes), "accounting": "measured_logical_state"},
        {"component": "queue_at_observed_max_depth", "bytes": 0, "accounting": "depth_was_zero; byte attribution unavailable"},
        {"component": "cuda_vram_peak_separate_from_rss", "bytes": int(streaming.vram_peak_bytes), "accounting": "measured_device_memory_not_subtracted_from_rss"},
        {"component": "unattributed_process_rss_residual", "bytes": residual, "accounting": "difference only; includes Python/PyTorch/allocator/model/other overhead"},
    ])
    atomic_csv(PROFILING / "rss_component_accounting.csv", frame)
    return frame


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    PROFILING.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    copy_table("table_supervised_baselines", "table_main_supervised")
    copy_table("table_graph_anomaly_baselines", "table_unsupervised_gad")
    copy_table("table_routing_flip_statistics", "table_routing_flip_statistics")
    copy_table("table_cross_chain", "table_cross_chain_strict")
    copy_table("table_streaming_100k", "table_integrated_streaming")

    policy = build_routing_policy()
    save_table("table_routing_policy", policy)
    save_table("table_accuracy_risk_cost_frontier", build_frontier(policy))

    gate = json.loads((ROOT / "cascade/acceptance_gate.json").read_text(encoding="utf-8"))
    cascade = pd.read_csv(ROOT / "cascade/cascade_summary.csv")
    cascade["tabular_cascade_gate"] = gate["gate"]
    cascade["fairness_scope"] = "production GIN only; incompatible tabular cascade claims removed"
    save_table("table_cascade_fairness", cascade)

    lpp_score = pd.read_csv(MANUSCRIPT_TABLES / "table_lpp_score_equivalence.csv")
    lpp_resource = pd.read_csv(MANUSCRIPT_TABLES / "table_lpp_resource.csv")
    pooled = lpp_score[lpp_score.chain == "pooled"].iloc[0]
    no_lpp = lpp_resource[lpp_resource.variant == "no_purge"].iloc[0]
    with_lpp = lpp_resource[lpp_resource.variant == "full_lpp_ttl_lru"].iloc[0]
    lpp = pd.DataFrame([{
        "scope": "pooled", "max_abs_score_diff": pooled.max_abs_score_diff_mean,
        "mean_abs_score_diff": pooled.mean_abs_score_diff_mean,
        "p99_abs_score_diff": pooled.p99_abs_score_diff_mean,
        "decision_disagreement_rate": pooled.decision_disagreement_rate_mean,
        "hash_equal": pooled.all_prediction_hash_equal,
        "peak_RSS_without_LPP_MB": no_lpp.peak_rss_mb,
        "peak_RSS_with_LPP_MB": with_lpp.peak_rss_mb,
        "throughput_without_LPP": no_lpp.throughput_events_s,
        "throughput_with_LPP": with_lpp.throughput_events_s,
        "state_growth_without_LPP_MB_per_10k": no_lpp.memory_slope_mb_per_10k,
        "state_growth_with_LPP_MB_per_10k": with_lpp.memory_slope_mb_per_10k,
        "allowed_claim": "decision-equivalent within evaluated thresholds; not bitwise-identical",
    }])
    save_table("table_lpp_equivalence_resource", lpp)

    hyperparameters = pd.read_csv(ROOT / "reproducibility/hyperparameters.csv")
    save_table("table_hyperparameters", hyperparameters)
    environment = json.loads((ROOT / "reproducibility/environment.json").read_text(encoding="utf-8"))
    environment_rows = []
    for key, value in environment.items():
        if key not in {"checkpoint_hashes", "config_hashes"}:
            environment_rows.append({"field": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value})
    save_table("table_environment", pd.DataFrame(environment_rows))
    save_table("table_calibration", build_calibration_table())

    figure_aliases = {
        "figure_risk_coverage.pdf": "fig_risk_coverage.pdf",
        "figure_mc_sensitivity.pdf": "fig_mc_sensitivity.pdf",
        "figure_reliability.pdf": "fig_reliability_diagram.pdf",
    }
    for source, destination in figure_aliases.items():
        shutil.copyfile(MANUSCRIPT_FIGURES / source, FIGURES / destination)
    build_risk_figures()
    build_streaming_figure()
    build_rss_accounting()

    claim = pd.read_csv(ROOT / "statistics/claim_status.csv").iloc[0]
    compile_report = json.loads((ROOT / "validation/tectonic_compile_validation.json").read_text(encoding="utf-8"))
    env = json.loads((ROOT / "reproducibility/environment.json").read_text(encoding="utf-8"))
    config_hash = env["config_hashes"]["configs/sci_v3_submission_r2/closure.yaml"]
    runtime = pd.read_csv(ROOT / "runtime/five_repeat_policy_summary.csv").set_index("policy").loc["validation_calibrated_dual"]
    revision = {
        "version": 2,
        "claims": [
            {
                "claim_id": "C-GIN-CASCADE-F1", "claim_text_template": "observed mean F1 difference",
                "metric": "delta_f1", "formatted_value": f"{claim.mean_delta_f1:+.3f}", "raw_value": claim.mean_delta_f1,
                "ci_low": claim.ci_low, "ci_high": claim.ci_high, "p_value": None, "adjusted_p_value": None,
                "significance_status": claim.status,
                "sample_support": {"n_per_seed": 3648, "n_positive_per_seed": 161, "n_negative_per_seed": 3487, "seeds": 5},
                "provenance": "measured_prediction", "canonical_source": "statistics/claim_status.csv",
                "raw_source": "cascade/predictions/ProductionLevel1GIN__seed*.csv", "config_sha256": config_hash,
                "git_sha": env["git_commit"],
            },
            {
                "claim_id": "C-RUNTIME-REPEATED-POLICY", "claim_text_template": "mean latency reduction vs full deep",
                "metric": "latency_reduction_pct", "formatted_value": f"{runtime.latency_reduction_vs_full_deep_pct:.2f}%",
                "raw_value": runtime.latency_reduction_vs_full_deep_pct, "ci_low": None, "ci_high": None,
                "p_value": None, "adjusted_p_value": None, "significance_status": "DESCRIPTIVE_MEASURED",
                "sample_support": {"events_per_repeat": 500, "runtime_positive_support": 0, "seeds": 5, "repeats_per_seed": 5},
                "provenance": "measured_e2e", "canonical_source": "runtime/five_repeat_policy_summary.csv",
                "raw_source": "runtime/five_repeat_policy_timing.csv", "config_sha256": config_hash,
                "git_sha": env["git_commit"],
            },
        ],
        "compiled_pdf_sha256": compile_report["pdf_sha256"],
    }
    atomic_json(ROOT / "revision_manifest.json", revision)

    required = [
        TABLES / "table_main_supervised.tex", TABLES / "table_unsupervised_gad.tex",
        TABLES / "table_routing_policy.tex", TABLES / "table_routing_flip_statistics.tex",
        TABLES / "table_cascade_fairness.tex", TABLES / "table_accuracy_risk_cost_frontier.tex",
        TABLES / "table_cross_chain_strict.tex", TABLES / "table_integrated_streaming.tex",
        TABLES / "table_lpp_equivalence_resource.tex", TABLES / "table_hyperparameters.tex",
        TABLES / "table_environment.tex", FIGURES / "fig_risk_coverage.pdf",
        FIGURES / "fig_fnr_coverage.pdf", FIGURES / "fig_mc_sensitivity.pdf",
        FIGURES / "fig_reliability_diagram.pdf", FIGURES / "fig_streaming_rss_throughput.pdf",
        ROOT / "revision_manifest.json", PROFILING / "rss_component_accounting.csv",
    ]
    validation = {
        "status": "PASS" if all(path.exists() for path in required) and compile_report["status"] == "PASS" else "FAIL",
        "compile_status": compile_report["status"],
        "required_deliverables": [{"path": str(path), "exists": path.exists()} for path in required],
        "scientific_gates": {"tabular_cascade": gate["gate"], "gin_claim": claim.status},
    }
    atomic_json(ROOT / "evidence_validation.json", validation)
    print(json.dumps({"status": validation["status"], "tables": len(list(TABLES.glob('*.tex'))), "figures": len(list(FIGURES.glob('*.pdf')))}, indent=2))


if __name__ == "__main__":
    main()
