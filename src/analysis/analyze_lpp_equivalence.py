#!/usr/bin/env python3
"""Quantify score, decision, hash, and ranking equivalence for the LPP variant."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.sci_v3_final_common import atomic_csv


def score_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def stable_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(values.size)
    return ranks


def run(prediction_dir: Path, metrics_path: Path, output_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    thresholds = metrics[metrics.model == "DLG-Full-Fusion"].set_index(["chain", "seed"])["threshold"].to_dict()
    rows: list[dict] = []
    for base_path in sorted(prediction_dir.glob("*__DLG-Full-Fusion__seed*.csv")):
        stem = base_path.stem
        chain = stem.split("__", 1)[0]
        seed = int(stem.rsplit("seed", 1)[1])
        lpp_path = prediction_dir / base_path.name.replace("DLG-Full-Fusion__", "DLG-Full-Fusion-LPP__")
        if not lpp_path.exists():
            continue
        base = pd.read_csv(base_path)
        lpp = pd.read_csv(lpp_path)
        merged = base.merge(lpp, on=["sample_id", "label"], suffixes=("_base", "_lpp"), validate="one_to_one")
        left = merged.score_base.to_numpy(dtype=float)
        right = merged.score_lpp.to_numpy(dtype=float)
        difference = np.abs(left - right)
        threshold = float(thresholds[(chain, seed)])
        disagreement = (left >= threshold) != (right >= threshold)
        rank_changes = stable_ranks(left) != stable_ranks(right)
        hash_equal = score_hash(left) == score_hash(right)
        rows.append(
            {
                "chain": chain,
                "seed": seed,
                "threshold": threshold,
                "n_samples": len(merged),
                "max_abs_score_diff": float(difference.max()),
                "mean_abs_score_diff": float(difference.mean()),
                "p95_abs_score_diff": float(np.quantile(difference, 0.95)),
                "p99_abs_score_diff": float(np.quantile(difference, 0.99)),
                "decision_disagreement_count": int(disagreement.sum()),
                "decision_disagreement_rate": float(disagreement.mean()),
                "prediction_hash_equal": bool(hash_equal),
                "ranking_order_change_count": int(rank_changes.sum()),
                "ranking_definition": "samples_with_changed_stable_rank",
                "allowed_claim": "bitwise-identical" if hash_equal else ("decision-equivalent within the evaluated threshold" if not disagreement.any() else "not decision-equivalent"),
            }
        )
    result = pd.DataFrame(rows)
    atomic_csv(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", default="results/results_sci_v2/main/predictions")
    parser.add_argument("--metrics", default="results/results_sci_v2/paper_eligible_results_long.csv")
    parser.add_argument("--output", default="results/sci_v3_final/statistics/lpp_equivalence.csv")
    args = parser.parse_args()
    frame = run(Path(args.prediction_dir), Path(args.metrics), Path(args.output))
    print(json.dumps({"records": len(frame), "decision_disagreements": int(frame.decision_disagreement_count.sum())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
