"""Derived 12-Dataset Extended Benchmark Integration and Complete-Case Analysis (Phase J).

Reads frozen Round 5 raw CSV (strictly read-only) and Defense extension raw CSV,
merges into derived benchmark_12dataset_view.csv, and computes complete-case
statistical rank comparisons without altering primary 10-dataset conclusions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCALABLE_MODELS = ["CoLA", "DOMINANT", "OCGNN", "DLG-Base", "DLG-Aug"]


def build_12dataset_view(round5_raw_path: Path, defense_raw_path: Path, output_dir: Path) -> pd.DataFrame:
    """Read-only merge of frozen Round 5 raw results and Defense extension results."""
    if not round5_raw_path.exists():
        raise FileNotFoundError(f"Frozen Round 5 raw CSV not found at {round5_raw_path}")
    if not defense_raw_path.exists():
        raise FileNotFoundError(f"Defense raw CSV not found at {defense_raw_path}")

    # Read Round 5 frozen raw
    r5_df = pd.read_csv(round5_raw_path)
    if "benchmark_origin" not in r5_df.columns:
        r5_df["benchmark_origin"] = "round5_primary"
    if "validation_f1" in r5_df.columns and "f1" not in r5_df.columns:
        r5_df["f1"] = r5_df["validation_f1"]

    # Read Defense extension raw
    def_df = pd.read_csv(defense_raw_path)
    if "benchmark_origin" not in def_df.columns:
        def_df["benchmark_origin"] = "defense_external_extension"
    if "f1" in def_df.columns and "validation_f1" not in def_df.columns:
        def_df["validation_f1"] = def_df["f1"]

    # Align columns
    common_cols = [c for c in r5_df.columns if c in def_df.columns]
    combined = pd.concat([r5_df[common_cols], def_df[common_cols]], ignore_index=True)

    out_file = output_dir / "extended_analysis" / "benchmark_12dataset_view.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_file, index=False)

    sha256_hash = hashlib.sha256(out_file.read_bytes()).hexdigest()
    (output_dir / "extended_analysis" / "benchmark_12dataset_view.csv.sha256").write_text(sha256_hash + "\n", encoding="utf-8")

    log.info("Saved 12-dataset combined view to %s (%d rows, sha256=%s)", out_file, len(combined), sha256_hash)
    return combined


def compute_complete_case_rankings(df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """Compute average rankings across all datasets for scalable model subset."""
    # Seed-first aggregation
    agg = df.groupby(["dataset", "model"])[metric].mean().reset_index()
    pivot = agg.pivot(index="dataset", columns="model", values=metric)

    # Filter for scalable models
    valid_models = [m for m in SCALABLE_MODELS if m in pivot.columns]
    filtered = pivot[valid_models].dropna()

    # Ranks (1 = highest performance)
    ranks = filtered.apply(lambda row: rankdata(-row, method="average"), axis=1, result_type="expand")
    ranks.columns = valid_models

    avg_ranks = ranks.mean().reset_index()
    avg_ranks.columns = ["model", f"{metric}_average_rank"]
    avg_ranks = avg_ranks.sort_values(by=f"{metric}_average_rank")
    return avg_ranks


def run_12dataset_statistical_tests(combined_df: pd.DataFrame, output_dir: Path) -> None:
    """Execute Friedman and Wilcoxon signed-rank tests with Holm correction on 12 datasets."""
    metrics = ["roc_auc", "pr_auc", "f1"]
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. 10-Dataset vs 12-Dataset Scalable Ranking Stability Table
    r5_df = combined_df[combined_df["benchmark_origin"] == "round5_primary"]
    
    ranking_records = []
    for m in SCALABLE_MODELS:
        row = {"model": m}
        for metric in metrics:
            r10 = compute_complete_case_rankings(r5_df, metric=metric)
            r12 = compute_complete_case_rankings(combined_df, metric=metric)
            
            rank10 = r10.loc[r10["model"] == m, f"{metric}_average_rank"].values
            rank12 = r12.loc[r12["model"] == m, f"{metric}_average_rank"].values
            
            row[f"10_Dataset_{metric.upper()}_Rank"] = round(float(rank10[0]), 2) if len(rank10) > 0 else np.nan
            row[f"12_Dataset_{metric.upper()}_Rank"] = round(float(rank12[0]), 2) if len(rank12) > 0 else np.nan
        ranking_records.append(row)

    ranking_df = pd.DataFrame(ranking_records)
    ranking_df.to_csv(tables_dir / "03_10_vs_12_dataset_scalable_ranking.csv", index=False)

    # 2. Friedman Test on 12 datasets
    friedman_results = []
    for metric in metrics:
        agg = combined_df.groupby(["dataset", "model"])[metric].mean().reset_index()
        pivot = agg.pivot(index="dataset", columns="model", values=metric)[SCALABLE_MODELS].dropna()
        
        stat, p_val = friedmanchisquare(*[pivot[col] for col in SCALABLE_MODELS])
        friedman_results.append({
            "metric": metric.upper(),
            "n_datasets": len(pivot),
            "n_models": len(SCALABLE_MODELS),
            "friedman_stat": round(float(stat), 4),
            "p_value": float(p_val),
            "significant_at_05": p_val < 0.05,
        })

    f_df = pd.DataFrame(friedman_results)
    f_df.to_csv(tables_dir / "04_12dataset_friedman_tests.csv", index=False)

    # 3. Pairwise Wilcoxon Signed-Rank Tests (DLG-Aug vs Others) with Holm step-down correction
    wilcoxon_records = []
    for metric in metrics:
        agg = combined_df.groupby(["dataset", "model"])[metric].mean().reset_index()
        pivot = agg.pivot(index="dataset", columns="model", values=metric)[SCALABLE_MODELS].dropna()
        
        aug_vals = pivot["DLG-Aug"].values
        comparisons = [m for m in SCALABLE_MODELS if m != "DLG-Aug"]
        
        raw_p_values = []
        stats = []
        for other in comparisons:
            other_vals = pivot[other].values
            diff = aug_vals - other_vals
            if np.all(diff == 0):
                w_stat, p = 0.0, 1.0
            else:
                w_stat, p = wilcoxon(aug_vals, other_vals)
            stats.append(w_stat)
            raw_p_values.append(p)
            
        # Holm-Bonferroni correction
        order = np.argsort(raw_p_values)
        m_comp = len(comparisons)
        adj_p_values = np.zeros(m_comp)
        
        for i, idx in enumerate(order):
            adj_p = min(1.0, raw_p_values[idx] * (m_comp - i))
            adj_p_values[idx] = adj_p

        # Enforce monotonicity
        for i in range(1, m_comp):
            idx = order[i]
            prev_idx = order[i - 1]
            if adj_p_values[idx] < adj_p_values[prev_idx]:
                adj_p_values[idx] = adj_p_values[prev_idx]

        for other, w_stat, p_raw, p_adj in zip(comparisons, stats, raw_p_values, adj_p_values):
            wilcoxon_records.append({
                "metric": metric.upper(),
                "comparison": f"DLG-Aug vs {other}",
                "w_statistic": round(float(w_stat), 2),
                "p_raw": float(p_raw),
                "p_holm_adj": float(p_adj),
                "significant_at_05": p_adj < 0.05,
            })

    w_df = pd.DataFrame(wilcoxon_records)
    w_df.to_csv(tables_dir / "05_12dataset_pairwise_wilcoxon_holm.csv", index=False)

    log.info("12-dataset extended statistical tests complete.")


def main():
    parser = argparse.ArgumentParser(description="Build 12-dataset extended view and statistics.")
    parser.add_argument("--round5-raw", type=str, default="outputs/sci_round5_final/raw/benchmark_raw.csv")
    parser.add_argument("--defense-raw", type=str, default="outputs/sci_defense_extension/raw/benchmark_raw.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/sci_defense_extension")
    args = parser.parse_args()

    r5_path = Path(args.round5_raw)
    def_path = Path(args.defense_raw)
    out_dir = Path(args.output_dir)

    combined_df = build_12dataset_view(r5_path, def_path, out_dir)
    run_12dataset_statistical_tests(combined_df, out_dir)
    print("12-Dataset extended view and complete-case analysis complete.")


if __name__ == "__main__":
    main()
