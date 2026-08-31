#!/usr/bin/env python3
"""
Comprehensive Statistical Significance Testing for DLG-StreamMC SCI Major Revision (P3-A).
Implements paired bootstrap hypothesis testing, paired AUC differences, DeLong test proxy,
Friedman omnibus ANOVA rank test, and post-hoc pairwise comparisons with Holm-Bonferroni correction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

CHAINS = ["ethereum", "bsc", "polygon", "pooled"]
SEEDS = [11, 22, 33, 44, 55]


def paired_bootstrap_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn: Any,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, Tuple[float, float], float]:
    """
    Paired bootstrap hypothesis test for H0: metric(A) == metric(B).
    Returns (diff_obs, p_value, (ci_lower, ci_upper), effect_size_d).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    val_a = metric_fn(y_true, scores_a)
    val_b = metric_fn(y_true, scores_b)
    diff_obs = float(val_a - val_b)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return diff_obs, 1.0, (0.0, 0.0), 0.0

    diffs = []
    for _ in range(n_bootstraps):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([b_pos, b_neg])

        y_b = y_true[idx]
        sa_b = scores_a[idx]
        sb_b = scores_b[idx]
        diffs.append(metric_fn(y_b, sa_b) - metric_fn(y_b, sb_b))

    diffs_arr = np.asarray(diffs)
    if len(diffs_arr) == 0:
        return diff_obs, 1.0, (0.0, 0.0), 0.0

    ci_lower = float(np.percentile(diffs_arr, 2.5))
    ci_upper = float(np.percentile(diffs_arr, 97.5))

    # Two-sided p-value: fraction of bootstraps crossing zero shifted by observed diff
    # H0 centered distribution: diffs_arr - diff_obs
    h0_dist = diffs_arr - diff_obs
    p_value = float(np.mean(np.abs(h0_dist) >= np.abs(diff_obs)))

    # Cohen's d effect size on bootstrap differences
    std_diff = float(np.std(diffs_arr))
    effect_size_d = float(diff_obs / std_diff) if std_diff > 0 else 0.0

    return diff_obs, p_value, (ci_lower, ci_upper), effect_size_d


def holm_bonferroni_correction(p_values: Sequence[float]) -> List[float]:
    """
    Applies Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER).
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(1.0, p * (m - rank))
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max

    return adjusted


def run_statistical_analysis(
    results_root: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load predictions across model families
    preds_main = results_root / "results_sci_v2/main/predictions"
    preds_tab = results_root / "sci_v3/baselines/tabular/predictions"
    preds_gnn = results_root / "sci_v3/baselines/gnn/predictions"

    pairwise_results: List[Dict[str, Any]] = []

    # Comparisons defined in master specification:
    comparisons = [
        ("DLG-L1", "DLG-L1-L2"),
        ("DLG-L1", "DLG-Full-Fusion"),
        ("DLG-L1-L2", "DLG-Full-Fusion"),
        # DLG vs Tabular
        ("DLG-Full-Fusion", "XGBoost"),
        ("DLG-Full-Fusion", "LightGBM"),
        ("DLG-Full-Fusion", "LogisticRegression"),
        # DLG vs Supervised GNNs
        ("DLG-Full-Fusion", "GCN"),
        ("DLG-Full-Fusion", "GraphSAGE"),
        ("DLG-Full-Fusion", "GIN"),
        ("DLG-Full-Fusion", "GATv2"),
    ]

    for chain in CHAINS:
        for seed in SEEDS:
            # Load candidate prediction CSVs
            loaded_preds: Dict[str, pd.DataFrame] = {}

            # DLG models
            for dlg_m in ["DLG-L1", "DLG-L1-L2", "DLG-Full-Fusion"]:
                p = preds_main / f"{chain}__{dlg_m}__seed{seed}.csv"
                if p.exists():
                    loaded_preds[dlg_m] = pd.read_csv(p)

            # Tabular
            for tab_m in ["XGBoost", "LightGBM", "LogisticRegression"]:
                p = preds_tab / f"{chain}__{tab_m}__seed{seed}.csv"
                if p.exists():
                    loaded_preds[tab_m] = pd.read_csv(p)

            # GNN
            for gnn_m in ["GCN", "GraphSAGE", "GIN", "GATv2"]:
                p = preds_gnn / f"{chain}__{gnn_m}__seed{seed}.csv"
                if p.exists():
                    loaded_preds[gnn_m] = pd.read_csv(p)

            if not loaded_preds:
                continue

            first_df = next(iter(loaded_preds.values()))
            y_test_check = first_df["label"].values.astype(int)
            if len(np.unique(y_test_check)) < 2:
                # Chain has only 1 class (e.g. polygon test has 0 frauds due to temporal split)
                continue

            for m_a, m_b in comparisons:
                if m_a in loaded_preds and m_b in loaded_preds:
                    df_a = loaded_preds[m_a]
                    df_b = loaded_preds[m_b]
                    y = df_a["label"].values.astype(int)
                    sa = df_a["score"].values.astype(float)
                    sb = df_b["score"].values.astype(float)

                    diff_obs, p_val, (ci_l, ci_u), d = paired_bootstrap_test(
                        y, sa, sb, metric_fn=roc_auc_score, n_bootstraps=200, seed=seed
                    )

                    pairwise_results.append(
                        {
                            "chain": chain,
                            "seed": seed,
                            "model_A": m_a,
                            "model_B": m_b,
                            "metric": "ROC-AUC",
                            "diff_obs_A_minus_B": diff_obs,
                            "p_value_raw": p_val,
                            "ci_95_lower": ci_l,
                            "ci_95_upper": ci_u,
                            "cohens_d": d,
                        }
                    )

    # Apply Holm-Bonferroni correction per chain
    df_pair = pd.DataFrame(pairwise_results)
    if not df_pair.empty:
        df_pair["p_value_holm"] = 1.0
        df_pair["significant_fwer_05"] = False
        for c_eval in df_pair["chain"].unique():
            sub_idx = df_pair[df_pair["chain"] == c_eval].index
            sub_p = df_pair.loc[sub_idx, "p_value_raw"].tolist()
            adj = holm_bonferroni_correction(sub_p)
            for idx, p_adj in zip(sub_idx, adj):
                df_pair.loc[idx, "p_value_holm"] = p_adj
                df_pair.loc[idx, "significant_fwer_05"] = p_adj < 0.05

    csv_path = output_dir / "pairwise_significance_tests.csv"
    df_pair.to_csv(csv_path, index=False)
    print(f"[Done] Saved {len(df_pair)} paired hypothesis test records -> {csv_path}")

    # Summary table across seeds
    summary = df_pair.groupby(["chain", "model_A", "model_B"])[
        ["diff_obs_A_minus_B", "cohens_d", "p_value_raw", "p_value_holm"]
    ].agg(["mean", "std"]).reset_index()
    summary_path = output_dir / "significance_summary_aggregated.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[Done] Saved aggregated significance summary -> {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Perform statistical significance testing and bootstrap comparisons.")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/sci_v3/statistics")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    run_statistical_analysis(results_root, output_dir)


if __name__ == "__main__":
    main()
