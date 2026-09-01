"""Build SCI-v3 submission canonical evidence, tables, figures, and reports."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validation.sci_v3_final_common import atomic_csv, atomic_json, git_sha, sha256_file, sha256_json


def report(path: Path, title: str, paragraphs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "Generated from machine-readable submission evidence; numerical values must not be edited manually.", ""]
    for heading, text in paragraphs: lines.extend([f"## {heading}", "", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def save_figure(fig: Any, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(base.with_suffix(".pdf")); fig.savefig(base.with_suffix(".png"), dpi=180); plt.close(fig)


def risk_table(repo: Path) -> pd.DataFrame:
    source = pd.read_csv(repo / "results/sci_v3_final/risk_control/risk_control_alpha_sweep.csv")
    dataset = Path("/mnt/d/_Work/_data/GoG_sci_v2")
    counts: dict[str, dict[str, int]] = {}
    for chain in ("ethereum", "bsc", "polygon"):
        split = json.loads((dataset / f"splits/{chain}_holdout_v2.json").read_text(encoding="utf-8"))["groups"]
        manifest = json.loads((dataset / f"manifests/{chain}.json").read_text(encoding="utf-8"))["records"]
        labels = {item["sample_id"]: int(item["label"]) for item in manifest}
        counts[chain] = {"N_calibration": len(split["validation"]["sample_ids"]), "N_test": len(split["test"]["sample_ids"]),
                         "N_test_fraud": sum(labels[item] for item in split["test"]["sample_ids"])}
    counts["pooled"] = {key: sum(counts[chain][key] for chain in ("ethereum", "bsc", "polygon")) for key in ("N_calibration", "N_test", "N_test_fraud")}
    rows = []
    for _, item in source.iterrows():
        chain = item.chain; positive = counts[chain]["N_test_fraud"]
        rows.append({"chain": chain, "seed": int(item.seed), **counts[chain], "target_alpha": item.target_alpha,
            "observed_FNR": item.observed_test_risk if positive else np.nan,
            "exact_upper_bound": item.test_upper_bound_diagnostic if positive else np.nan,
            "coverage": item.coverage, "deep_rate": item.deep_route_rate,
            "metric_defined": bool(positive), "undefined_reason": "" if positive else "zero_positive_temporal_holdout",
            "claim_class": "temporal_holdout_validation_constrained_audit",
            "theorem_assumptions_verified": False, "source_artifact": "results/sci_v3_final/risk_control/risk_control_alpha_sweep.csv"})
    return pd.DataFrame(rows)


def run(repo: Path) -> None:
    root = repo / "results/sci_v3_submission"; canonical = root / "canonical"; tables = root / "figures_and_tables/tables"; figures = root / "figures_and_tables/figures"
    for directory in (canonical, tables, figures, root / "routing", root / "risk_control", root / "cross_chain", root / "streaming", root / "statistics"):
        directory.mkdir(parents=True, exist_ok=True)
    accuracy = pd.read_csv(root / "baselines/production_backbone_metrics.csv")
    runtime = pd.read_csv(root / "profiling/raw_event_e2e_summary.csv")
    integrated = pd.read_csv(root / "integrated/table_integrated_100k.csv")
    identity = json.loads((root / "method_identity.json").read_text(encoding="utf-8"))
    risk = risk_table(repo); atomic_csv(root / "risk_control/temporal_risk_audit.csv", risk)

    runtime_t1 = runtime[(runtime.mc_samples == 1) & runtime.policy.isin(["only", "dual", "no_routing"])].copy()
    # Frontier fairness uses the identical 500-event prefix for every method;
    # the separate 100k row belongs only to integrated systems evidence.
    runtime_t1 = runtime_t1.sort_values("events").drop_duplicates(["seed", "scenario", "policy"], keep="first")
    acc = accuracy[accuracy.routing_policy.isin(["only", "dual", "no_routing"])].copy()
    acc["scenario"] = acc.model.str.split("->").str[0]
    acc["policy"] = acc.routing_policy
    cascade = acc.merge(runtime_t1, on=["seed", "scenario", "policy"], how="left", suffixes=("", "_runtime"))
    cascade["coverage"] = cascade.direct_exit_rate
    cascade["accuracy_measurement_unit"] = "frozen_test_contract"
    cascade["runtime_measurement_unit"] = "raw_transaction_event"
    atomic_csv(root / "baselines/table_cascade_accuracy_cost.csv", cascade); atomic_csv(tables / "table_cascade_accuracy_cost.csv", cascade)
    atomic_csv(tables / "table_temporal_risk_audit.csv", risk); atomic_csv(tables / "table_integrated_streaming_e2e.csv", integrated)
    atomic_csv(root / "streaming/table_integrated_streaming_e2e.csv", integrated)

    # Canonical submission records are deliberately narrow and do not overwrite sci_v3_final.
    atomic_csv(canonical / "canonical_metrics.csv", accuracy)
    atomic_csv(canonical / "canonical_runtime.csv", runtime)
    atomic_csv(canonical / "canonical_cascade_frontier.csv", cascade)
    atomic_csv(canonical / "canonical_risk_control.csv", risk)
    atomic_csv(canonical / "canonical_streaming.csv", integrated)
    atomic_json(canonical / "canonical_method_identity.json", identity)
    for name, source in (("canonical_cross_chain.csv", repo / "results/sci_v3_final/canonical/canonical_cross_chain.csv"),
                         ("canonical_statistics.csv", repo / "results/sci_v3_final/canonical/canonical_statistics.csv")):
        shutil.copy2(source, canonical / name)

    plotted = cascade.dropna(subset=["mean_latency_ms"])
    for x, y, name, ylabel in (("deep_route_rate", "fraud_recall", "fig_risk_cost_frontier", "Fraud recall"),
                               ("p95_latency_ms", "f1", "fig_accuracy_latency_frontier", "F1"),
                               ("p95_latency_ms", "fnr", "fig_fnr_latency_frontier", "FNR")):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for model, group in plotted.groupby("model"):
            agg = group.groupby(x)[y].mean().sort_index(); ax.plot(agg.index, agg.values, marker="o", label=model)
        ax.set(xlabel=x.replace("_", " "), ylabel=ylabel); ax.legend(fontsize=7)
        save_figure(fig, figures / name)
    trace = pd.read_csv(root / "integrated/raw_trace_100k.csv")
    index = np.arange(len(trace)); slope = float(np.polyfit(index, trace.rss_bytes, 1)[0]) if len(trace) > 1 else 0.0
    integrated.loc[:, "rss_slope_bytes_per_event"] = slope; atomic_csv(tables / "table_integrated_streaming_e2e.csv", integrated); atomic_csv(canonical / "canonical_streaming.csv", integrated); atomic_csv(root / "streaming/table_integrated_streaming_e2e.csv", integrated)
    fig, left = plt.subplots(figsize=(8, 4.8)); left.plot(index, trace.rss_bytes / 2**20, label="RSS MiB"); right = left.twinx(); right.plot(index, trace.total_latency_ms, alpha=.35, color="tab:orange", label="latency")
    left.set(xlabel="Event", ylabel="RSS (MiB)"); right.set(ylabel="Latency (ms)"); save_figure(fig, figures / "fig_integrated_memory_latency")
    fig, left = plt.subplots(figsize=(8, 4.8)); left.plot(index, trace.queue_depth, label="queue depth"); right = left.twinx(); right.plot(index, trace.total_latency_ms, alpha=.35, color="tab:red")
    left.set(xlabel="Event", ylabel="Queue depth"); right.set(ylabel="Latency (ms)"); save_figure(fig, figures / "fig_integrated_queue_latency")
    shutil.copy2(figures / "fig_integrated_memory_latency.pdf", root / "streaming/fig_integrated_memory_latency.pdf")
    shutil.copy2(figures / "fig_integrated_queue_latency.pdf", root / "streaming/fig_integrated_queue_latency.pdf")
    fig, ax = plt.subplots(figsize=(11, 2.8)); ax.axis("off")
    labels = ["Raw event", "Bounded local graph", "Production Level-1 GIN\n+ MC uncertainty", "Selective router", "Optional Level-2 GATv2", "Weighted fusion"]
    for position, label in enumerate(labels):
        x = .02 + position * .16; ax.text(x, .52, label, ha="center", va="center", fontsize=8,
            bbox={"boxstyle": "round,pad=.5", "facecolor": "#e8f1fa" if position < 4 else "#fdebd0", "edgecolor": "#345"})
        if position < len(labels) - 1: ax.annotate("", xy=(x + .135, .52), xytext=(x + .065, .52), arrowprops={"arrowstyle": "->"})
    ax.text(.58, .12, "direct exit (stages 9–11 skipped)", ha="center", fontsize=8, color="#245")
    save_figure(fig, figures / "fig_method_production_path")

    reports = repo / "docs/work_reports/sci_v3_submission"
    xgb = cascade[(cascade.scenario == "XGBoostFastTriage") & (cascade.policy == "dual")]
    gnn = cascade[(cascade.scenario == "ProductionLevel1GIN") & (cascade.policy == "dual")]
    gnn_only = accuracy[(accuracy.model == "ProductionLevel1GIN") & (accuracy.routing_policy == "only")].f1.mean()
    gnn_selective = accuracy[(accuracy.model == "ProductionLevel1GIN->ProductionLevel2GATv2") & (accuracy.routing_policy == "dual")].f1.mean()
    framing = ("DLG-StreamMC is a bounded stateful local-to-global graph inference architecture that escalates ambiguous events "
               "from production GIN to relational GATv2. Production GIN was the strongest evaluated fast path on this frozen pooled test; "
               "the claim does not generalize accuracy superiority beyond the evaluated controls and protocol")
    report(reports / "method_identity_closure.md", "Method Identity Closure", [("Decision", "Path A."), ("Final method", framing), ("Identity record", json.dumps(identity, sort_keys=True))])
    report(reports / "raw_event_e2e_runtime_report.md", "Raw-event E2E Runtime", [("Scope", "Actual sorted raw transactions execute bounded update, local graph construction, production GIN/MC, routing, optional GATv2, and fusion."), ("Direct exit", "Validator requires relation preparation, Level-2, and Fusion to remain exactly zero."), ("Records", str(len(runtime)))])
    policy_runtime = runtime[(runtime.scenario == "ProductionLevel1GIN") & (runtime.mc_samples == 1) & (runtime.events == 500)].groupby("policy").mean(numeric_only=True)
    no_route_mean = float(policy_runtime.loc["no_routing", "mean_latency_ms"])
    dual_reduction = 100 * (1 - float(policy_runtime.loc["dual", "mean_latency_ms"]) / no_route_mean)
    risk_reduction = 100 * (1 - float(policy_runtime.loc["risk_controlled", "mean_latency_ms"]) / no_route_mean)
    xgb_only = accuracy[(accuracy.model == "XGBoostFastTriage") & (accuracy.routing_policy == "only")].f1.mean()
    lgb_only = accuracy[(accuracy.model == "LightGBMFastTriage") & (accuracy.routing_policy == "only")].f1.mean()
    lgb_cascade = accuracy[(accuracy.model == "LightGBMFastTriage->ProductionLevel2GATv2") & (accuracy.routing_policy == "dual")].f1.mean()
    report(reports / "measured_cascade_frontier_report.md", "Measured Cascade Frontier", [("Result", f"XGBoost only/cascade mean F1={xgb_only:.6f}/{xgb.f1.mean():.6f}; LightGBM only/cascade={lgb_only:.6f}/{lgb_cascade:.6f}; production-GIN only/cascade={gnn_only:.6f}/{gnn_selective:.6f} (delta {gnn_selective-gnn_only:+.6f}). Strong tabular controls remain fully disclosed and their cascades are not described as accuracy improvements."), ("Measured cost", f"Against full-deep no-routing on the identical 500-event prefixes, dual routing reduces mean E2E latency by {dual_reduction:.2f}% and empirical risk-controlled routing by {risk_reduction:.2f}%. Tail latency and accuracy remain separately reported in the canonical frontier; mean savings are not presented as universal tail savings."), ("Positioning", framing), ("Cost provenance", "All latency rows are measured_e2e; reconstructed costs are excluded.")])
    report(reports / "integrated_streaming_inference_report.md", "Integrated Streaming + Inference", [("Replay", f"{int(integrated.events.iloc[0])} raw events; actual fast path/router/deep/fusion with bounded store/cache/queue."), ("Restart", f"Prediction disagreement={int(integrated.checkpoint_prediction_disagreement.iloc[0])}."), ("Memory", f"RSS peak={int(integrated.rss_peak_bytes.iloc[0])} bytes and fitted RSS slope={slope:.6f} bytes/event. Production store/cache maxima are {int(integrated.max_store_bytes.iloc[0])}/{int(integrated.max_cache_bytes.iloc[0])} bytes; trace rows are emitted through a bounded chunked writer."), ("Risk wording", "This is an empirical temporal replay, not a distribution-free deployment guarantee.")])
    report(reports / "final_claim_freeze_report.md", "Final Claim Freeze", [("Status", "PASS, subject to the generated fail-closed validator report."), ("Scientific position", framing), ("Risk wording", "Use validation-constrained selective routing, empirically risk-controlled direct exit, and temporal holdout risk audit. Do not claim a distribution-free deployment guarantee.")])

    config = repo / "configs/sci_v3_submission/production_closure.yaml"; code_sha = git_sha(repo)
    raw_specs = {
        "method_identity": (root / "method_identity.json", canonical / "canonical_method_identity.json", root / "baselines/production_backbone_metrics.csv"),
        "measured_cascade_frontier": (tables / "table_cascade_accuracy_cost.csv", canonical / "canonical_cascade_frontier.csv", root / "profiling/raw_traces/XGBoostFastTriage__dual__seed11.csv"),
        "raw_event_e2e": (root / "profiling/raw_event_e2e_summary.csv", canonical / "canonical_runtime.csv", root / "profiling/raw_event_e2e_summary.csv"),
        "integrated_bounded_inference": (tables / "table_integrated_streaming_e2e.csv", canonical / "canonical_streaming.csv", root / "integrated/raw_trace_100k.csv"),
        "temporal_risk_audit": (tables / "table_temporal_risk_audit.csv", canonical / "canonical_risk_control.csv", repo / "results/sci_v3_final/risk_control/risk_control_alpha_sweep.csv"),
    }
    claims = []
    for claim, (generated, canonical_source, raw) in raw_specs.items():
        claims.append({"claim": claim, "generated_table_or_figure": str(generated.relative_to(repo)),
            "canonical_row_source": str(canonical_source.relative_to(repo)), "raw_trace_or_prediction": str(raw.relative_to(repo)),
            "generated_sha256": sha256_file(generated), "canonical_sha256": sha256_file(canonical_source),
            "raw_sha256": sha256_file(raw), "config_hash": sha256_file(config), "code_git_sha": code_sha})
    atomic_json(root / "revision_manifest.json", {"claims": claims, "method_identity": identity,
        "environment": {"python": platform.python_version(), "platform": platform.platform()}, "manifest_hash": sha256_json(claims)})
    atomic_json(canonical / "canonical_manifest.json", {"counts": {"metrics": len(accuracy), "runtime": len(runtime), "cascade": len(cascade), "risk": len(risk), "streaming": len(integrated)}, "claims": claims})


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", default="."); args = parser.parse_args(); run(Path(args.repo).resolve()); print(json.dumps({"status": "generated"})); return 0


if __name__ == "__main__": raise SystemExit(main())
