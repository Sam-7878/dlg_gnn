#!/usr/bin/env python3
"""
Aggregate Round 5 frozen benchmark (10 datasets, 355 runs) + Defense Extension Round D3 (2 datasets, 80 runs)
into unified 12-dataset statistical view and Markdown/CSV reports.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROUND5_RAW_PATH = Path("outputs/sci_round5_final/raw/benchmark_raw.csv")
DEFENSE_RAW_PATH = Path("outputs/sci_defense_extension_real/benchmark/benchmark_raw.csv")
OUTPUT_TABLES_DIR = Path("outputs/sci_defense_extension_real/tables")
DOCS_DIR = Path("docs/work_reports/210_Defense_Extension_Round_D3")

MODELS = ["DOMINANT", "AnomalyDAE", "CoLA", "CONAD", "GADNR", "OCGNN", "DLG-Base", "DLG-Aug"]
ALL_12_DATASETS = [
    "Amazon", "BitcoinOTC", "CiteSeer", "Cora", "DGraphFin",
    "Elliptic", "Flickr", "PubMed", "Reddit", "Yelp",
    "DARPA-TC-THEIA", "LANL-RedTeam"
]


def load_all_runs() -> pd.DataFrame:
    dfs = []
    if ROUND5_RAW_PATH.exists():
        df_r5 = pd.read_csv(ROUND5_RAW_PATH)
        df_r5["source"] = "round5_frozen"
        dfs.append(df_r5)
        log.info("Loaded Round 5 frozen runs: %d rows", len(df_r5))
    else:
        log.warning("Round 5 raw path not found at %s", ROUND5_RAW_PATH)

    if DEFENSE_RAW_PATH.exists():
        df_def = pd.read_csv(DEFENSE_RAW_PATH)
        df_def["source"] = "defense_d3_real"
        dfs.append(df_def)
        log.info("Loaded Defense Extension D3 runs: %d rows", len(df_def))
    else:
        log.warning("Defense raw path not found at %s", DEFENSE_RAW_PATH)

    if not dfs:
        raise FileNotFoundError("No benchmark runs found to aggregate.")

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def build_unified_metric_tables(df: pd.DataFrame):
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Filter to success runs
    success_df = df[df["status"] == "success"].copy()
    
    metrics = ["roc_auc", "pr_auc", "f1", "fit_seconds"]
    
    for m in metrics:
        if m not in success_df.columns:
            continue
        pivot_mean = success_df.pivot_table(
            index="dataset", columns="model", values=m, aggfunc="mean"
        ).reindex(index=ALL_12_DATASETS, columns=MODELS)
        
        pivot_std = success_df.pivot_table(
            index="dataset", columns="model", values=m, aggfunc="std"
        ).reindex(index=ALL_12_DATASETS, columns=MODELS).fillna(0.0)

        # Formatted Mean ± Std table
        formatted = pd.DataFrame(index=ALL_12_DATASETS, columns=MODELS)
        for d in ALL_12_DATASETS:
            for mod in MODELS:
                mean_val = pivot_mean.loc[d, mod] if d in pivot_mean.index and mod in pivot_mean.columns else np.nan
                std_val = pivot_std.loc[d, mod] if d in pivot_std.index and mod in pivot_std.columns else np.nan
                if pd.isna(mean_val):
                    formatted.loc[d, mod] = "—"
                else:
                    if m == "fit_seconds":
                        formatted.loc[d, mod] = f"{mean_val:.1f} ± {std_val:.1f}s"
                    else:
                        formatted.loc[d, mod] = f"{mean_val:.4f} ± {std_val:.4f}"

        csv_path = OUTPUT_TABLES_DIR / f"unified_12_dataset_{m}_table.csv"
        formatted.to_csv(csv_path)
        log.info("Saved metric table: %s", csv_path)

    # Combined master summary CSV
    agg_summary = success_df.groupby(["dataset", "model"]).agg({
        "roc_auc": ["mean", "std", "min", "max", "count"],
        "pr_auc": ["mean", "std"],
        "f1": ["mean", "std"],
        "fit_seconds": ["mean", "std"],
    }).reset_index()
    summary_path = OUTPUT_TABLES_DIR / "unified_12_dataset_statistical_summary.csv"
    agg_summary.to_csv(summary_path, index=False)
    log.info("Saved master statistical summary: %s", summary_path)


def main():
    log.info("Aggregating 12-dataset benchmark results...")
    df = load_all_runs()
    build_unified_metric_tables(df)
    log.info("Statistical aggregation completed successfully.")


if __name__ == "__main__":
    main()
