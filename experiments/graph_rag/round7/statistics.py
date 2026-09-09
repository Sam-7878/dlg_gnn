"""Frozen paired significance protocols for Round 7 model comparisons."""
from __future__ import annotations

import numpy as np


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    positives = int(ranked.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def _mean_difference(labels: np.ndarray, left: list[np.ndarray], right: list[np.ndarray]) -> float:
    return float(np.mean([
        average_precision(labels, left_scores) - average_precision(labels, right_scores)
        for left_scores, right_scores in zip(left, right)
    ]))


def paired_panel_comparison(
    labels: np.ndarray,
    left_panel: list[np.ndarray],
    right_panel: list[np.ndarray],
    *,
    n_resamples: int = 10_000,
    seed: int = 20260903,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    left = [np.asarray(scores, dtype=float) for scores in left_panel]
    right = [np.asarray(scores, dtype=float) for scores in right_panel]
    if len(left) != len(right) or not left:
        raise ValueError("paired panels must have the same non-zero run count")
    if any(scores.shape != labels.shape for scores in left + right):
        raise ValueError("all prediction arrays must align with labels")
    observed = _mean_difference(labels, left, right)
    rng = np.random.default_rng(seed)
    ordinary = np.empty(n_resamples, dtype=float)
    stratified = np.empty(n_resamples, dtype=float)
    randomized = np.empty(n_resamples, dtype=float)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    for index in range(n_resamples):
        draw = rng.integers(0, len(labels), len(labels))
        ordinary[index] = _mean_difference(labels[draw], [x[draw] for x in left], [x[draw] for x in right])
        stratified_draw = np.concatenate((
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ))
        rng.shuffle(stratified_draw)
        stratified[index] = _mean_difference(
            labels[stratified_draw], [x[stratified_draw] for x in left], [x[stratified_draw] for x in right],
        )
        permuted_differences = []
        for left_scores, right_scores in zip(left, right):
            swap = rng.random(len(labels)) < 0.5
            permuted_left = np.where(swap, right_scores, left_scores)
            permuted_right = np.where(swap, left_scores, right_scores)
            permuted_differences.append(
                average_precision(labels, permuted_left) - average_precision(labels, permuted_right)
            )
        randomized[index] = float(np.mean(permuted_differences))
    ordinary_valid = ordinary[np.isfinite(ordinary)]
    if ordinary_valid.size == 0:
        raise RuntimeError("ordinary bootstrap produced no samples containing both classes")
    return {
        "mean_auc_pr_difference": observed,
        "ordinary_ci95_low": float(np.quantile(ordinary_valid, 0.025)),
        "ordinary_ci95_high": float(np.quantile(ordinary_valid, 0.975)),
        "ordinary_probability_gt_zero": float(np.mean(ordinary_valid > 0)),
        "stratified_ci95_low": float(np.quantile(stratified, 0.025)),
        "stratified_ci95_high": float(np.quantile(stratified, 0.975)),
        "stratified_probability_gt_zero": float(np.mean(stratified > 0)),
        "randomization_p_value_two_sided": float(
            (1 + np.sum(np.abs(randomized) >= abs(observed))) / (n_resamples + 1)
        ),
        "n_bootstrap": n_resamples,
        "n_bootstrap_valid": int(ordinary_valid.size),
        "n_class_stratified_bootstrap": n_resamples,
        "n_randomization": n_resamples,
        "n_events": len(labels),
        "n_positive": int(labels.sum()),
        "paired_runs": len(left),
    }

