#!/usr/bin/env python3
"""
Analyze routing decisions and sample-level flips between Level 1 and Fusion models.
P0-A Correctness Audit for DLG-StreamMC SCI Major Revision.
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

from gog_fraud.evaluation.routing_metrics import (
    FlipMetrics,
    SampleRoutingTrace,
    calculate_entropy,
    evaluate_routing_traces,
    traces_to_dataframe,
)
from gog_fraud.selection.router import SelectiveRouter, TriageOutput

CHAINS = ["ethereum", "bsc", "polygon", "pooled"]
SEEDS = [11, 22, 33, 44, 55]


def load_threshold(df_records: pd.DataFrame, chain: str, seed: int, model: str = "DLG-Full-Fusion") -> float:
    sub = df_records[(df_records["chain"] == chain) & (df_records["seed"] == seed) & (df_records["model"] == model)]
    if len(sub) == 0:
        # Fallback to general threshold search
        sub = df_records[(df_records["chain"] == chain) & (df_records["seed"] == seed)]
    if len(sub) == 0:
        return 0.5
    return float(sub["threshold"].values[0])


def run_flip_analysis(
    results_root: Path,
    output_dir: Path,
    save_traces: bool = True,
) -> pd.DataFrame:
    records_path = results_root / "paper_eligible_results_long.csv"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing eligible results records at {records_path}")

    df_records = pd.read_csv(records_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"
    if save_traces:
        traces_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    for chain in CHAINS:
        for seed in SEEDS:
            f_l1 = results_root / "main/predictions" / f"{chain}__DLG-L1__seed{seed}.csv"
            f_l2 = results_root / "main/predictions" / f"{chain}__DLG-L1-L2__seed{seed}.csv"
            f_ff = results_root / "main/predictions" / f"{chain}__DLG-Full-Fusion__seed{seed}.csv"
            f_mc = results_root / "mc/predictions" / f"{chain}__seed{seed}__T8.csv"

            if not (f_l1.exists() and f_l2.exists() and f_ff.exists() and f_mc.exists()):
                print(f"[Warning] Incomplete prediction files for {chain} seed {seed}. Skipping.")
                continue

            df_l1 = pd.read_csv(f_l1)
            df_l2 = pd.read_csv(f_l2)
            df_ff = pd.read_csv(f_ff)
            df_mc = pd.read_csv(f_mc)

            threshold = load_threshold(df_records, chain, seed, "DLG-Full-Fusion")
            margin = 0.15
            tau_b = max(0.0, threshold - margin)
            tau_f = min(1.0, threshold + margin)

            policies = {
                "no_routing": SelectiveRouter(tau_b=0.0, tau_f=1.0, tau_u=0.0, threshold_version="v3-no"),
                "variance_only": SelectiveRouter(tau_b=0.0, tau_f=1.0, tau_u=0.001, threshold_version="v3-var"),
                "dual_threshold": SelectiveRouter(tau_b=tau_b, tau_f=tau_f, tau_u=0.001, threshold_version="v3-dual"),
                "risk_sensitive": SelectiveRouter(
                    tau_b=tau_b, tau_f=tau_f, tau_u=0.001, tau_r=max(threshold, 0.5), threshold_version="v3-risk"
                ),
            }

            for policy_name, router in policies.items():
                routes = []
                reasons = []

                for mean, var in zip(df_mc["score"].values, df_mc["mc_variance"].values):
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

                traces, metrics = evaluate_routing_traces(
                    sample_ids=df_l1["sample_id"].tolist(),
                    labels=df_l1["label"].tolist(),
                    l1_scores=df_l1["score"].tolist(),
                    mc_means=df_mc["score"].tolist(),
                    mc_vars=df_mc["mc_variance"].tolist(),
                    routes=routes,
                    route_reasons=reasons,
                    l2_scores=df_l2["score"].tolist(),
                    fusion_scores=df_ff["score"].tolist(),
                    threshold=threshold,
                    chain=chain,
                    seed=seed,
                    policy=policy_name,
                    num_mc_samples=8,
                )

                if save_traces:
                    trace_df = traces_to_dataframe(traces)
                    trace_out = traces_dir / f"traces__{chain}__seed{seed}__{policy_name}.parquet"
                    trace_df.to_parquet(trace_out, index=False)

                row = {
                    "chain": chain,
                    "seed": seed,
                    "policy": policy_name,
                    "threshold": threshold,
                    **metrics.to_dict(),
                }
                summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    csv_out = output_dir / "routing_flips.csv"
    df_summary.to_csv(csv_out, index=False)
    print(f"[Done] Saved {len(df_summary)} routing flip records to {csv_out}")
    return df_summary


def main():
    parser = argparse.ArgumentParser(description="Analyze routing decisions and prediction flips.")
    parser.add_argument("--results-root", type=str, default="results/results_sci_v2", help="Root directory of results.")
    parser.add_argument("--output-dir", type=str, default="results/sci_v3", help="Output directory for flip metrics.")
    parser.add_argument("--no-traces", action="store_true", help="Skip saving individual sample trace files.")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    df_summary = run_flip_analysis(results_root, output_dir, save_traces=not args.no_traces)

    # Print summary of pooled seed averages
    pooled = df_summary[df_summary["chain"] == "pooled"]
    agg = pooled.groupby("policy")[
        ["deep_route_rate", "n_deep", "n_flips", "wrong_to_correct", "correct_to_wrong", "net_gain", "overall_f1", "overall_recall"]
    ].mean()
    print("\n--- Pooled Policies (5-Seed Average) ---")
    print(agg.to_string())


if __name__ == "__main__":
    main()
