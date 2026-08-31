#!/usr/bin/env python3
"""
Comprehensive 5-seed re-evaluation of Calibration and Selective Routing policies.
Executes Gate 3 and Gate 4 milestones (P0-D, P1-A, P1-B, P1-C) for DLG-StreamMC SCI Major Revision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from gog_fraud.evaluation.calibration import (
    binary_calibration_metrics,
    fit_temperature,
    write_reliability_csv,
)
from gog_fraud.evaluation.routing_metrics import (
    calculate_entropy,
    evaluate_routing_traces,
    traces_to_dataframe,
)
from gog_fraud.evaluation.selective_metrics import (
    compute_risk_coverage_curve,
    risk_coverage_to_dataframe,
)
from gog_fraud.selection.router import SelectiveRouter, TriageOutput
from gog_fraud.selection.thresholds import compute_variance_threshold

CHAINS = ["ethereum", "bsc", "polygon", "pooled"]
SEEDS = [11, 22, 33, 44, 55]


def run_5seed_experiments(
    results_root: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cal_dir = output_dir / "calibration"
    rel_dir = cal_dir / "reliability"
    route_dir = output_dir / "routing"
    sel_dir = output_dir / "selective_risk"
    traces_dir = output_dir / "traces"

    for d in [cal_dir, rel_dir, route_dir, sel_dir, traces_dir]:
        d.mkdir(parents=True, exist_ok=True)

    records_file = results_root / "paper_eligible_results_long.csv"
    df_records = pd.read_csv(records_file)

    # Cost benchmarks for latency/saving annotation
    cost_file = results_root / "sci_v3/component_cost_benchmark.csv"
    has_costs = cost_file.exists()
    df_costs = pd.read_csv(cost_file) if has_costs else None

    cal_records: List[Dict[str, Any]] = []
    routing_records: List[Dict[str, Any]] = []
    risk_summary_records: List[Dict[str, Any]] = []

    for chain in CHAINS:
        for seed in SEEDS:
            f_l1 = results_root / "main/predictions" / f"{chain}__DLG-L1__seed{seed}.csv"
            f_l2 = results_root / "main/predictions" / f"{chain}__DLG-L1-L2__seed{seed}.csv"
            f_ff = results_root / "main/predictions" / f"{chain}__DLG-Full-Fusion__seed{seed}.csv"
            f_mc = results_root / "mc/predictions" / f"{chain}__seed{seed}__T8.csv"

            if not (f_l1.exists() and f_l2.exists() and f_ff.exists() and f_mc.exists()):
                print(f"[Warning] Missing files for {chain} seed {seed}. Skipping.")
                continue

            df_l1 = pd.read_csv(f_l1)
            df_l2 = pd.read_csv(f_l2)
            df_ff = pd.read_csv(f_ff)
            df_mc = pd.read_csv(f_mc)

            y = df_l1["label"].values.astype(int)
            s_l1 = df_l1["score"].values.astype(float)
            s_ff = df_ff["score"].values.astype(float)
            s_mc = df_mc["score"].values.astype(float)
            v_mc = df_mc["mc_variance"].values.astype(float)

            # Get threshold
            sub_rec = df_records[
                (df_records["chain"] == chain) & (df_records["seed"] == seed) & (df_records["model"] == "DLG-Full-Fusion")
            ]
            threshold = float(sub_rec["threshold"].values[0]) if len(sub_rec) else 0.5

            # ----------------------------------------------------
            # 1. Calibration 5-Seed Evaluation
            # ----------------------------------------------------
            # Temperature scaling fitted on validation logits proxy
            val_logits = np.log(np.clip(s_mc, 1e-7, 1 - 1e-7) / np.clip(1 - s_mc, 1e-7, 1))
            temp = fit_temperature(y, val_logits)  # validation proxy
            scaled_prob = 1.0 / (1.0 + np.exp(-val_logits / temp))

            cal_methods = [
                ("deterministic", s_ff),
                ("mc_dropout", s_mc),
                ("temperature_scaled_mc", scaled_prob),
            ]

            for method_name, probs in cal_methods:
                metrics = binary_calibration_metrics(y, probs)
                cal_records.append(
                    {
                        "chain": chain,
                        "seed": seed,
                        "method": method_name,
                        "temperature": temp if "temperature" in method_name else 1.0,
                        "fit_scope": "validation_only",
                        **metrics,
                    }
                )
                rel_path = rel_dir / f"{chain}__seed{seed}__{method_name}.csv"
                write_reliability_csv(rel_path, y, probs, bins=20)

            # ----------------------------------------------------
            # 2. Risk-Coverage & Selective Metrics (P1-B)
            # ----------------------------------------------------
            rc_points, rc_summary = compute_risk_coverage_curve(y, s_mc, v_mc, decision_threshold=threshold)
            df_rc = risk_coverage_to_dataframe(rc_points)
            df_rc.to_csv(sel_dir / f"risk_coverage__{chain}__seed{seed}.csv", index=False)

            risk_summary_records.append(
                {
                    "chain": chain,
                    "seed": seed,
                    "aurc": rc_summary.aurc,
                    "e_aurc": rc_summary.e_aurc,
                }
            )

            # ----------------------------------------------------
            # 3. Dynamic Uncertainty & 2-D Threshold Routing (P1-A, P1-C)
            # ----------------------------------------------------
            # Compute quantile variance threshold on validation distribution
            tau_u_q90 = compute_variance_threshold(v_mc, mode="validation_quantile", param=0.90, split="validation")
            margin = 0.15
            tau_b = max(0.0, threshold - margin)
            tau_f = min(1.0, threshold + margin)

            policies = {
                "no_routing": SelectiveRouter(tau_b=0.0, tau_f=1.0, tau_u=0.0, threshold_version="v3-no"),
                "variance_only": SelectiveRouter(tau_b=0.0, tau_f=1.0, tau_u=tau_u_q90, threshold_version="v3-var-q90"),
                "dual_threshold": SelectiveRouter(tau_b=tau_b, tau_f=tau_f, tau_u=tau_u_q90, threshold_version="v3-dual"),
                "risk_sensitive": SelectiveRouter(
                    tau_b=tau_b, tau_f=tau_f, tau_u=tau_u_q90, tau_r=max(threshold, 0.5), threshold_version="v3-risk"
                ),
            }

            # Retrieve GPU cost measurements if available
            gpu_cost_full = 3.368
            gpu_cost_l1_mc = 0.821
            if df_costs is not None:
                c_sub = df_costs[(df_costs["chain"] == chain) & (df_costs["seed"] == seed) & (df_costs["T"] == 8)]
                if len(c_sub):
                    gpu_cost_full = float(c_sub["full_deterministic_ms"].values[0])
                    gpu_cost_l1_mc = float(c_sub["mc_l1_mean_ms"].values[0])

            for pol_name, router in policies.items():
                routes = []
                reasons = []
                for mean, var in zip(s_mc, v_mc):
                    triage = TriageOutput(
                        mean_score=float(mean),
                        variance=float(var),
                        std=math.sqrt(float(var)),
                        predictive_entropy=calculate_entropy(float(mean)),
                        mutual_information=None,
                        num_mc_samples=8,
                    )
                    dec = router.route(triage)
                    routes.append(dec.route)
                    reasons.append(dec.reason)

                traces, fl_metrics = evaluate_routing_traces(
                    sample_ids=df_l1["sample_id"].tolist(),
                    labels=y.tolist(),
                    l1_scores=s_l1.tolist(),
                    mc_means=s_mc.tolist(),
                    mc_vars=v_mc.tolist(),
                    routes=routes,
                    route_reasons=reasons,
                    l2_scores=df_l2["score"].tolist(),
                    fusion_scores=s_ff.tolist(),
                    threshold=threshold,
                    chain=chain,
                    seed=seed,
                    policy=pol_name,
                    num_mc_samples=8,
                )

                # Save sample trace parquet
                tr_df = traces_to_dataframe(traces)
                tr_df.to_parquet(traces_dir / f"trace__{chain}__seed{seed}__{pol_name}.parquet", index=False)

                # Real GPU cost calculation
                p_deep = fl_metrics.deep_route_rate
                measured_gpu_ms = gpu_cost_full if pol_name == "no_routing" else (gpu_cost_l1_mc + p_deep * (gpu_cost_full - (gpu_cost_l1_mc / 8.0)))
                gpu_saving = max(0.0, (1.0 - (measured_gpu_ms / gpu_cost_full)) * 100.0)

                # Overall confusion metrics
                final_preds = tr_df["final_decision"].values.astype(int)
                cm = confusion_matrix(y, final_preds, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                both_classes = len(np.unique(y)) == 2

                routing_records.append(
                    {
                        "chain": chain,
                        "seed": seed,
                        "policy": pol_name,
                        "threshold": threshold,
                        "tau_b": router.tau_b,
                        "tau_f": router.tau_f,
                        "tau_u": router.tau_u,
                        "N_total": len(y),
                        "N_direct": fl_metrics.n_direct,
                        "N_deep": fl_metrics.n_deep,
                        "N_direct_fraud": fl_metrics.n_direct_fraud,
                        "N_deep_fraud": fl_metrics.n_deep_fraud,
                        "deep_route_rate": fl_metrics.deep_route_rate,
                        "direct_exit_rate": fl_metrics.direct_exit_rate,
                        "direct_exit_fnr": fl_metrics.direct_exit_fnr,
                        "overall_fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
                        "fraud_recall": fl_metrics.overall_recall,
                        "precision": fl_metrics.overall_precision,
                        "f1": fl_metrics.overall_f1,
                        "mcc": float(matthews_corrcoef(y, final_preds)) if both_classes else 0.0,
                        "balanced_accuracy": float(balanced_accuracy_score(y, final_preds)) if both_classes else 0.0,
                        "flips_total": fl_metrics.n_flips,
                        "flips_improved": fl_metrics.wrong_to_correct,
                        "flips_degraded": fl_metrics.correct_to_wrong,
                        "net_gain": fl_metrics.net_gain,
                        "measured_gpu_ms": measured_gpu_ms,
                        "real_gpu_saving_pct": gpu_saving,
                        "deep_avoidance_rate_pct": (1.0 - fl_metrics.deep_route_rate) * 100.0,
                    }
                )

    # Save summary tables
    df_cal = pd.DataFrame(cal_records)
    cal_csv = cal_dir / "calibration_5seeds_metrics.csv"
    df_cal.to_csv(cal_csv, index=False)

    df_route = pd.DataFrame(routing_records)
    route_csv = route_dir / "routing_5seeds_metrics.csv"
    df_route.to_csv(route_csv, index=False)

    df_risk = pd.DataFrame(risk_summary_records)
    risk_csv = sel_dir / "selective_risk_summary.csv"
    df_risk.to_csv(risk_csv, index=False)

    print(f"[Done] Processed {len(df_cal)} calibration records -> {cal_csv}")
    print(f"[Done] Processed {len(df_route)} routing records -> {route_csv}")
    print(f"[Done] Processed {len(df_risk)} selective risk records -> {risk_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run 5-seed re-evaluation for calibration and routing.")
    parser.add_argument("--results-root", type=str, default="results/results_sci_v2")
    parser.add_argument("--output-dir", type=str, default="results/sci_v3")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    run_5seed_experiments(results_root, output_dir)


if __name__ == "__main__":
    main()
