"""Defense-Specific Analysis and Topology Metric Computation (Round D1).

Generates Defense Table 1 (Performance), Table 2 (Scalability), DLG component deltas,
and defense-specific topology metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import t
import torch

from gog_fraud.extensions.defense.defense_registry import DEFENSE_DATASETS, load_defense_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def compute_defense_topology(dataset_name: str, data) -> dict:
    """Compute exact fraud/security-aware topology metrics on defense graph."""
    x = data.x.detach().cpu().numpy()
    edge_index = data.edge_index.detach().cpu().numpy()
    y = data.y.detach().cpu().numpy().astype(np.int64)

    n_nodes = len(y)
    n_edges = edge_index.shape[1]
    n_pos = int((y == 1).sum())
    n_neg = n_nodes - n_pos
    pos_ratio = n_pos / n_nodes if n_nodes > 0 else 0.0

    src = edge_index[0]
    dst = edge_index[1]

    # Edge homophily
    same_label = (y[src] == y[dst])
    raw_edge_homophily = float(same_label.mean()) if n_edges > 0 else 1.0

    # Positive-conditioned homophily (edges originating from positive nodes)
    pos_src_mask = (y[src] == 1)
    if pos_src_mask.sum() > 0:
        pos_conditioned_homophily = float((y[dst][pos_src_mask] == 1).mean())
        pos_to_normal_mixing = float((y[dst][pos_src_mask] == 0).mean())
    else:
        pos_conditioned_homophily = 0.0
        pos_to_normal_mixing = 0.0

    # Normal-conditioned homophily
    neg_src_mask = (y[src] == 0)
    if neg_src_mask.sum() > 0:
        norm_conditioned_homophily = float((y[dst][neg_src_mask] == 0).mean())
        norm_to_pos_mixing = float((y[dst][neg_src_mask] == 1).mean())
    else:
        norm_conditioned_homophily = 1.0
        norm_to_pos_mixing = 0.0

    # Degree statistics
    degrees = np.bincount(src, minlength=n_nodes) + np.bincount(dst, minlength=n_nodes)
    avg_degree = float(degrees.mean())

    return {
        "dataset": dataset_name,
        "nodes": n_nodes,
        "edges": n_edges,
        "features": x.shape[1],
        "positives": n_pos,
        "positive_ratio": round(pos_ratio, 6),
        "raw_edge_homophily": round(raw_edge_homophily, 4),
        "pos_conditioned_homophily": round(pos_conditioned_homophily, 4),
        "norm_conditioned_homophily": round(norm_conditioned_homophily, 4),
        "pos_to_normal_mixing": round(pos_to_normal_mixing, 4),
        "norm_to_pos_mixing": round(norm_to_pos_mixing, 4),
        "avg_degree": round(avg_degree, 2),
    }


def analyze_defense_benchmark(raw_csv_path: Path, output_dir: Path) -> None:
    """Analyze defense benchmark raw CSV and output summary tables."""
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found at {raw_csv_path}")

    raw = pd.read_csv(raw_csv_path)
    success_raw = raw[raw["status"] == "success"].copy()

    # 1. Seed-level aggregated performance summary
    summary_rows = []
    metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall", "mcc", "fit_seconds", "peak_rss_mb", "peak_vram_mb"]

    for (dataset, model), group in success_raw.groupby(["dataset", "model"], sort=True):
        row = {"dataset": dataset, "model": model, "n_seeds": len(group)}
        for m in metrics:
            if m in group.columns:
                vals = group[m].astype(float)
                row[f"{m}_mean"] = float(vals.mean())
                row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                row[f"{m}_median"] = float(vals.median())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    sum_dir = output_dir / "summary"
    sum_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(sum_dir / "seed_aggregated_performance.csv", index=False)

    # 2. Defense Paper Table 1: Performance Summary
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    t1_rows = []
    for r in summary_rows:
        t1_rows.append({
            "dataset": r["dataset"],
            "model": r["model"],
            "ROC-AUC": f"{r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f}",
            "PR-AUC": f"{r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}",
            "Validation-Selected F1": f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}",
            "Fit Seconds": f"{r['fit_seconds_mean']:.2f}s",
        })
    t1_df = pd.DataFrame(t1_rows)
    t1_df.to_csv(table_dir / "01_defense_performance_summary.csv", index=False)

    # 3. DLG-Aug vs DLG-Base Delta Analysis
    dlg_deltas = []
    for d in DEFENSE_DATASETS:
        d_df = summary_df[summary_df["dataset"] == d]
        aug_row = d_df[d_df["model"] == "DLG-Aug"]
        base_row = d_df[d_df["model"] == "DLG-Base"]
        dom_row = d_df[d_df["model"] == "DOMINANT"]

        if not aug_row.empty and not base_row.empty:
            aug_f1 = aug_row["f1_mean"].values[0]
            base_f1 = base_row["f1_mean"].values[0]
            aug_pr = aug_row["pr_auc_mean"].values[0]
            base_pr = base_row["pr_auc_mean"].values[0]
            aug_roc = aug_row["roc_auc_mean"].values[0]
            base_roc = base_row["roc_auc_mean"].values[0]

            delta_f1 = aug_f1 - base_f1
            delta_pr = aug_pr - base_pr
            delta_roc = aug_roc - base_roc

            dlg_deltas.append({
                "dataset": d,
                "DLG-Base F1": round(base_f1, 4),
                "DLG-Aug F1": round(aug_f1, 4),
                "Delta F1 (Aug - Base)": round(delta_f1, 4),
                "Delta PR-AUC (Aug - Base)": round(delta_pr, 4),
                "Delta ROC-AUC (Aug - Base)": round(delta_roc, 4),
                "Effect Type": "Positive" if delta_f1 > 0.005 else ("Negative" if delta_f1 < -0.005 else "Near-Neutral")
            })

    delta_df = pd.DataFrame(dlg_deltas)
    delta_df.to_csv(table_dir / "02_dlg_augmentation_deltas.csv", index=False)

    # 4. Topology analysis
    topo_dir = output_dir / "topology"
    topo_dir.mkdir(parents=True, exist_ok=True)
    topo_records = []
    for d in DEFENSE_DATASETS:
        data = load_defense_dataset(d)
        topo = compute_defense_topology(d, data)
        topo_records.append(topo)
    topo_df = pd.DataFrame(topo_records)
    topo_df.to_csv(topo_dir / "defense_graph_topology.csv", index=False)

    log.info("Defense analysis complete. Tables and topology written to %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Analyze defense extension results.")
    parser.add_argument("--output-dir", type=str, default="outputs/sci_defense_extension")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    raw_csv = out_dir / "raw" / "benchmark_raw.csv"
    analyze_defense_benchmark(raw_csv, out_dir)
    print("Defense analysis completed successfully.")


if __name__ == "__main__":
    main()
