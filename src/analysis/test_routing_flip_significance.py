#!/usr/bin/env python3
"""Paired decision and metric tests for Level-1 versus routed final predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from validation.sci_v3_final_common import atomic_csv


def metric_pair(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    tp = int(((y == 1) & (prediction == 1)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    f1 = 2.0 * tp / max(1, 2 * tp + fp + fn)
    recall = tp / max(1, tp + fn)
    return f1, recall


def bootstrap_deltas(y: np.ndarray, before: np.ndarray, after: np.ndarray, seed: int, iterations: int) -> tuple[float, float, float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = y.size
    f1_before, recall_before = metric_pair(y, before)
    f1_after, recall_after = metric_pair(y, after)
    f1_values = np.empty(iterations, dtype=float)
    recall_values = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 100):
        count = min(100, iterations - start)
        sample = rng.integers(0, n, size=(count, n))
        sampled_y = y[sample]
        batch_f1 = []
        batch_recall = []
        for prediction in (after, before):
            sampled_prediction = prediction[sample]
            tp = ((sampled_y == 1) & (sampled_prediction == 1)).sum(axis=1)
            fp = ((sampled_y == 0) & (sampled_prediction == 1)).sum(axis=1)
            fn = ((sampled_y == 1) & (sampled_prediction == 0)).sum(axis=1)
            batch_f1.append(2.0 * tp / np.maximum(1, 2 * tp + fp + fn))
            batch_recall.append(tp / np.maximum(1, tp + fn))
        f1_values[start : start + count] = batch_f1[0] - batch_f1[1]
        recall_values[start : start + count] = batch_recall[0] - batch_recall[1]
    return (
        f1_after - f1_before,
        float(np.quantile(f1_values, 0.025)),
        float(np.quantile(f1_values, 0.975)),
        recall_after - recall_before,
        float(np.quantile(recall_values, 0.025)),
        float(np.quantile(recall_values, 0.975)),
    )


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def run(trace_dir: Path, output_path: Path, iterations: int) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(trace_dir.glob("trace__*__*.parquet")):
        frame = pd.read_parquet(path)
        if frame.empty or frame.policy.iloc[0] not in {"no_routing", "dual_threshold", "risk_sensitive"}:
            continue
        y = frame.label.to_numpy(dtype=int)
        before = frame.l1_decision.to_numpy(dtype=int)
        after = frame.final_decision.to_numpy(dtype=int)
        before_correct = before == y
        after_correct = after == y
        wrong_to_correct = int((~before_correct & after_correct).sum())
        correct_to_wrong = int((before_correct & ~after_correct).sum())
        discordant = wrong_to_correct + correct_to_wrong
        p_value = float(binomtest(min(wrong_to_correct, correct_to_wrong), discordant, 0.5).pvalue) if discordant else 1.0
        accuracy_delta = float(after_correct.mean() - before_correct.mean())
        f1_delta, f1_low, f1_high, recall_delta, recall_low, recall_high = bootstrap_deltas(
            y, before, after, int(frame.seed.iloc[0]), iterations
        )
        rows.append(
            {
                "chain": frame.chain.iloc[0],
                "seed": int(frame.seed.iloc[0]),
                "policy": frame.policy.iloc[0],
                "n_samples": len(frame),
                "discordant_pairs": discordant,
                "wrong_to_correct": wrong_to_correct,
                "correct_to_wrong": correct_to_wrong,
                "mcnemar_exact_p_value": p_value,
                "accuracy_delta": accuracy_delta,
                "accuracy_effect_per_1000": accuracy_delta * 1000.0,
                "f1_delta": f1_delta,
                "f1_delta_ci95_low": f1_low,
                "f1_delta_ci95_high": f1_high,
                "recall_delta": recall_delta,
                "recall_delta_ci95_low": recall_low,
                "recall_delta_ci95_high": recall_high,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_adjusted_p_value"] = holm(result.mcnemar_exact_p_value.tolist())
        result["significance_established_0_05"] = result.holm_adjusted_p_value < 0.05
    atomic_csv(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default="results/sci_v3/traces")
    parser.add_argument("--output", default="results/sci_v3_final/statistics/routing_flip_significance.csv")
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()
    frame = run(Path(args.trace_dir), Path(args.output), args.iterations)
    print(json.dumps({"records": len(frame), "significant": int(frame.get("significance_established_0_05", pd.Series(dtype=bool)).sum())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
