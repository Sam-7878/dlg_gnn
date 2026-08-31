#!/usr/bin/env python3
"""
GPU Cost & Latency Benchmark across inference components.
P0-B Compute Saving Remeasurement for DLG-StreamMC SCI Major Revision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from gog_fraud.pipelines.run_round4_experiments import (
    CHAINS,
    ContractDLG,
    SciV2Records,
    _data,
    _fit_dlg,
    _normalize,
)

T_VALUES = [1, 3, 5, 8, 10, 20, 30]


def benchmark_components(
    dataset_root: Path,
    output_dir: Path,
    device: torch.device,
    reps: int = 50,
    warmup: int = 15,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = SciV2Records(dataset_root)

    results: List[Dict[str, Any]] = []

    chains_to_test = ["ethereum", "bsc", "polygon", "pooled"]

    for chain in chains_to_test:
        train, valid, test = (dataset.ids(chain, g) for g in ("train", "validation", "test"))
        tx, ty = dataset.arrays(train)
        vx, vy = dataset.arrays(valid)
        qx, qy = dataset.arrays(test)
        tx, vx, qx = _normalize(tx, vx, qx)

        n_test = len(qx)
        data = _data(tx, qx).to(device)
        offset = len(tx)
        x, edge_index = data.x, data.edge_index

        for seed in [11, 22, 33]:
            model, _ = _fit_dlg(tx, ty, variant="DLG-Full-Fusion", seed=seed, epochs=50, device=device)
            model.eval()

            # Helper for accurate CUDA timing
            def measure(fn) -> tuple[float, float, float, float]:
                latencies = []
                for _ in range(warmup):
                    fn()
                if device.type == "cuda":
                    torch.cuda.synchronize()

                for _ in range(reps):
                    t0 = time.perf_counter()
                    fn()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - t0) * 1000.0)

                arr = np.asarray(latencies)
                return float(arr.mean()), float(np.median(arr)), float(np.percentile(arr, 95)), float(np.percentile(arr, 99))

            # 1. L1 deterministic (T=1)
            def run_l1():
                with torch.no_grad():
                    loc = model.local(x)
                    return model.local_head(loc)[offset:]

            # 2. L2 GNN only
            def run_l2():
                with torch.no_grad():
                    loc = model.local(x)
                    r = torch.relu(model.conv1(loc, edge_index))
                    r = torch.relu(model.conv2(r, edge_index))
                    return model.relation_head(r)[offset:]

            # 3. Fusion only (given local and relation)
            with torch.no_grad():
                loc_cached = model.local(x)
                r_cached = torch.relu(model.conv1(loc_cached, edge_index))
                r_cached = torch.relu(model.conv2(r_cached, edge_index))

            def run_fusion():
                with torch.no_grad():
                    gate = torch.sigmoid(model.gate(torch.cat((loc_cached, r_cached), dim=1))).view(-1)
                    return gate[offset:]

            # 4. Full Pipeline (L1 + L2 + Fusion)
            def run_full():
                with torch.no_grad():
                    return model(x, edge_index)[offset:]

            l1_mean, l1_med, l1_p95, l1_p99 = measure(run_l1)
            l2_mean, l2_med, l2_p95, l2_p99 = measure(run_l2)
            fus_mean, fus_med, fus_p95, fus_p99 = measure(run_fusion)
            full_mean, full_med, full_p95, full_p99 = measure(run_full)

            base_record = {
                "chain": chain,
                "seed": seed,
                "n_test": n_test,
                "l1_deterministic_ms": l1_mean,
                "l2_ms": l2_mean,
                "fusion_only_ms": fus_mean,
                "full_deterministic_ms": full_mean,
                "vram_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0,
                "vram_max_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0,
            }

            # Measure MC sweeps
            for T in T_VALUES:
                model.train(T > 1)  # activate dropout for MC

                def run_mc():
                    with torch.no_grad():
                        for _ in range(T):
                            loc = model.local(x)
                            _ = model.local_head(loc)[offset:]

                mc_mean, mc_med, mc_p95, mc_p99 = measure(run_mc)
                model.eval()

                rec = dict(base_record)
                rec.update(
                    {
                        "T": T,
                        "mc_l1_mean_ms": mc_mean,
                        "mc_l1_median_ms": mc_med,
                        "mc_l1_p95_ms": mc_p95,
                        "mc_l1_p99_ms": mc_p99,
                        "mc_throughput_nodes_sec": (n_test / (mc_mean / 1000.0)) if mc_mean > 0 else 0.0,
                    }
                )
                results.append(rec)

    df = pd.DataFrame(results)
    csv_out = output_dir / "component_cost_benchmark.csv"
    df.to_csv(csv_out, index=False)
    print(f"[Done] Saved {len(df)} component cost records to {csv_out}")
    return df


def evaluate_gate_a(df_costs: pd.DataFrame, df_flips: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Compare Cost_selective(T) against Cost_full using measured execution times:
    Cost_selective(T) = Cost_L1_MC(T) + P(deep) * (Cost_full - Cost_L1)
    Cost_full = Full deterministic pipeline
    """
    # Average across seeds
    cost_summary = df_costs.groupby(["chain", "T"])[
        ["l1_deterministic_ms", "l2_ms", "full_deterministic_ms", "mc_l1_mean_ms", "vram_max_mb"]
    ].mean().reset_index()

    # Get deep_route_rate per chain for dual_threshold
    dual_flips = df_flips[df_flips["policy"] == "dual_threshold"].groupby("chain")["deep_route_rate"].mean().to_dict()

    eval_rows = []
    for _, row in cost_summary.iterrows():
        c = row["chain"]
        T = int(row["T"])
        p_deep = dual_flips.get(c, 0.45)
        c_l1_mc = row["mc_l1_mean_ms"]
        c_full = row["full_deterministic_ms"]
        c_deep_extra = c_full - row["l1_deterministic_ms"]

        c_selective = c_l1_mc + p_deep * c_deep_extra
        saving_pct = (1.0 - (c_selective / c_full)) * 100.0
        gate_a_passed = bool(c_selective < c_full)

        eval_rows.append(
            {
                "chain": c,
                "T": T,
                "p_deep": p_deep,
                "cost_full_ms": c_full,
                "cost_l1_mc_ms": c_l1_mc,
                "cost_selective_ms": c_selective,
                "real_compute_saving_pct": saving_pct,
                "gate_a_passed": gate_a_passed,
                "deep_avoidance_rate_pct": (1.0 - p_deep) * 100.0,
            }
        )

    df_eval = pd.DataFrame(eval_rows)
    eval_csv = output_dir / "gate_a_cost_evaluation.csv"
    df_eval.to_csv(eval_csv, index=False)
    print(f"[Done] Saved Gate A evaluation to {eval_csv}")
    return df_eval


def main():
    parser = argparse.ArgumentParser(description="Benchmark component costs and evaluate Gate A.")
    parser.add_argument("--dataset-root", type=str, default="/mnt/d/_Work/_data/GoG_sci_v2")
    parser.add_argument("--flips-csv", type=str, default="results/sci_v3/routing_flips.csv")
    parser.add_argument("--output-dir", type=str, default="results/sci_v3")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    flips_csv = Path(args.flips_csv).resolve()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df_costs = benchmark_components(dataset_root, output_dir, device)
    df_flips = pd.read_csv(flips_csv)
    df_eval = evaluate_gate_a(df_costs, df_flips, output_dir)

    print("\n--- Gate A Cost Evaluation Summary (Pooled) ---")
    print(df_eval[df_eval["chain"] == "pooled"][["T", "cost_full_ms", "cost_selective_ms", "real_compute_saving_pct", "deep_avoidance_rate_pct", "gate_a_passed"]].to_string(index=False))


if __name__ == "__main__":
    main()
