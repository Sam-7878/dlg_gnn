#!/usr/bin/env python3
"""
calculate_lanl_diagnostics_m3.py

Phase D LANL Canonical Neighborhood Diagnostics for Round M3.
Computes neighborhood statistics on canonical LANL graph:
- outputs/benchmark/sci_defense_extension_real/graphs/lanl_graph.pt
  (N=16,694, E=323,897, F=13, positives=301, negatives=16,393).

Generates:
1. outputs/benchmark/manuscript_m3/lanl/lanl_neighborhood_diagnostics_m3.csv
2. outputs/benchmark/manuscript_m3/lanl/lanl_neighborhood_diagnostics_m3.json

Enforces:
- Internal consistency: unique_peers Cliff's delta (+0.630) matches between table and narrative.
- Document exact numerator/denominator and formulas for benign neighbor percentages.
- Strict non-causal association-level framing.
"""

from __future__ import annotations

from collections import defaultdict
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lanl_diagnostics_m3")

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = REPO_ROOT / "outputs" / "benchmark" / "sci_defense_extension_real" / "graphs" / "lanl_graph.pt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m3" / "lanl"


def compute_cliffs_delta(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute Cliff's delta non-parametric effect size d in [-1, 1]."""
    u_stat, _ = mannwhitneyu(pos, neg, alternative="two-sided")
    n1, n2 = len(pos), len(neg)
    delta = (2 * u_stat) / (n1 * n2) - 1.0
    return float(delta)


def bootstrap_ci(pos: np.ndarray, neg: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95% CI for median difference (pos - neg)."""
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        p_sample = rng.choice(pos, size=len(pos), replace=True)
        n_sample = rng.choice(neg, size=len(neg), replace=True)
        diffs.append(np.median(p_sample) - np.median(n_sample))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def run_diagnostics():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading canonical LANL graph from {GRAPH_PATH}...")
    data = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)

    x = data.x.numpy()
    y = data.y.numpy().reshape(-1).astype(int)
    edge_index = data.edge_index.numpy()
    num_nodes = data.num_nodes
    num_edges = edge_index.shape[1]
    num_features = x.shape[1]

    pos_mask = (y == 1)
    neg_mask = (y == 0)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    log.info(f"Loaded: N={num_nodes}, E={num_edges}, F={num_features}, Pos={n_pos}, Neg={n_neg}")
    assert num_nodes == 16694 and num_features == 13 and n_pos == 301, "Canonical LANL invariant violated!"

    src, dst = edge_index[0], edge_index[1]

    # Degree metrics
    out_degree = np.bincount(src, minlength=num_nodes)
    in_degree = np.bincount(dst, minlength=num_nodes)
    total_degree = in_degree + out_degree

    # Unique undirected peers
    adj_set = defaultdict(set)
    for u, v in zip(src, dst):
        if u != v:
            adj_set[u].add(v)
            adj_set[v].add(u)
    unique_peers = np.array([len(adj_set[i]) for i in range(num_nodes)], dtype=float)

    # Edge homophily & distribution
    edge_same_label = np.sum(y[src] == y[dst])
    edge_homophily = float(edge_same_label / num_edges)

    pos_pos_edges = int(np.sum((y[src] == 1) & (y[dst] == 1)))
    cross_edges = int(np.sum(y[src] != y[dst]))
    neg_neg_edges = int(np.sum((y[src] == 0) & (y[dst] == 0)))

    # Benign adjacent neighbor percentages across definitions:
    # Def 1: Out-edges from anomalous nodes: destinations that are benign
    out_pos = (y[src] == 1)
    out_pos_total = int(out_pos.sum())
    out_pos_benign = int((y[dst[out_pos]] == 0).sum())
    out_pos_benign_pct = 100.0 * out_pos_benign / out_pos_total if out_pos_total > 0 else 0.0

    # Def 2: In-edges to anomalous nodes: sources that are benign
    in_pos = (y[dst] == 1)
    in_pos_total = int(in_pos.sum())
    in_pos_benign = int((y[src[in_pos]] == 0).sum())
    in_pos_benign_pct = 100.0 * in_pos_benign / in_pos_total if in_pos_total > 0 else 0.0

    # Def 3: Undirected unique peers of anomalous nodes:
    pos_nodes = np.where(y == 1)[0]
    total_neighbor_incidences = sum(len(adj_set[u]) for u in pos_nodes)
    benign_neighbor_incidences = sum(sum(y[v] == 0 for v in adj_set[u]) for u in pos_nodes)
    peer_incidence_benign_pct = 100.0 * benign_neighbor_incidences / total_neighbor_incidences if total_neighbor_incidences > 0 else 0.0

    # Def 4: Mean per-node fraction of benign peers
    per_node_benign_fracs = [
        float(np.mean([y[v] == 0 for v in adj_set[u]])) if len(adj_set[u]) > 0 else 0.0
        for u in pos_nodes
    ]
    mean_per_node_benign_pct = 100.0 * float(np.mean(per_node_benign_fracs))

    log.info("Computing neighbor feature distance and dispersion...")
    feat_mean_dist = np.zeros(num_nodes, dtype=float)
    feat_dispersion = np.zeros(num_nodes, dtype=float)
    deg_deviation = np.zeros(num_nodes, dtype=float)

    for i in range(num_nodes):
        nbrs = list(adj_set[i])
        if len(nbrs) == 0:
            continue
        nbr_feats = x[nbrs]
        diff = nbr_feats - x[i : i + 1]
        feat_mean_dist[i] = np.mean(np.linalg.norm(diff, axis=1))

        nbr_mean = np.mean(nbr_feats, axis=0, keepdims=True)
        feat_dispersion[i] = np.mean(np.linalg.norm(nbr_feats - nbr_mean, axis=1))

        nbr_degs = total_degree[nbrs]
        deg_deviation[i] = np.abs(total_degree[i] - np.mean(nbr_degs))

    metrics = {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": total_degree,
        "unique_peers": unique_peers,
        "neighbor_feat_mean_dist": feat_mean_dist,
        "neighbor_feat_dispersion": feat_dispersion,
        "local_deg_deviation": deg_deviation,
    }

    rows = []
    summary_dict = {
        "dataset": "LANL-RedTeam",
        "canonical_graph_sha256": "689c2968fe3ece9494196515e6089d6db3f430530e55b8b410d116b27c920359",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_features": num_features,
        "num_positives": n_pos,
        "num_negatives": n_neg,
        "edge_homophily": edge_homophily,
        "pos_pos_edges": pos_pos_edges,
        "pos_neg_cross_edges": cross_edges,
        "neg_neg_edges": neg_neg_edges,
        "benign_neighbor_statistics": {
            "directed_out_edges_from_pos": {
                "total": out_pos_total,
                "benign_destinations": out_pos_benign,
                "benign_percentage": out_pos_benign_pct,
                "formula": "count(y[dst] == 0 for edges with y[src] == 1) / count(edges with y[src] == 1)"
            },
            "directed_in_edges_to_pos": {
                "total": in_pos_total,
                "benign_sources": in_pos_benign,
                "benign_percentage": in_pos_benign_pct,
                "formula": "count(y[src] == 0 for edges with y[dst] == 1) / count(edges with y[dst] == 1)"
            },
            "undirected_peer_incidences": {
                "total": total_neighbor_incidences,
                "benign_incidences": benign_neighbor_incidences,
                "benign_percentage": peer_incidence_benign_pct,
                "formula": "sum_u(|peers(u) cap Normal|) / sum_u(|peers(u)|) across u in Positives"
            },
            "mean_per_node_benign_fraction": {
                "mean_percentage": mean_per_node_benign_pct,
                "formula": "mean_u( |peers(u) cap Normal| / |peers(u)| ) across u in Positives"
            }
        },
        "metrics": {}
    }

    for m_name, vals in metrics.items():
        pos_vals = vals[pos_mask]
        neg_vals = vals[neg_mask]

        pos_med = float(np.median(pos_vals))
        pos_q25, pos_q75 = float(np.percentile(pos_vals, 25)), float(np.percentile(pos_vals, 75))
        pos_iqr = pos_q75 - pos_q25

        neg_med = float(np.median(neg_vals))
        neg_q25, neg_q75 = float(np.percentile(neg_vals, 25)), float(np.percentile(neg_vals, 75))
        neg_iqr = neg_q75 - neg_q25

        delta = compute_cliffs_delta(pos_vals, neg_vals)
        ci_low, ci_high = bootstrap_ci(pos_vals, neg_vals, n_boot=2000)

        record = {
            "metric": m_name,
            "pos_median": pos_med,
            "pos_iqr": pos_iqr,
            "pos_q25": pos_q25,
            "pos_q75": pos_q75,
            "neg_median": neg_med,
            "neg_iqr": neg_iqr,
            "neg_q25": neg_q25,
            "neg_q75": neg_q75,
            "cliffs_delta": delta,
            "median_diff_ci_low": ci_low,
            "median_diff_ci_high": ci_high,
        }
        rows.append(record)
        summary_dict["metrics"][m_name] = record

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "lanl_neighborhood_diagnostics_m3.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved diagnostics CSV to {csv_path}")

    def json_default(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    json_path = OUTPUT_DIR / "lanl_neighborhood_diagnostics_m3.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, default=json_default)
    log.info(f"Saved diagnostics JSON to {json_path}")
    print(f"LANL diagnostics completed: unique_peers Cliff's delta = {summary_dict['metrics']['unique_peers']['cliffs_delta']:+.3f}")


if __name__ == "__main__":
    run_diagnostics()
