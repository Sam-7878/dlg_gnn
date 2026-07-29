from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def paired_bootstrap_difference(y_true, score_a, score_b, metric: Callable, *, iterations: int = 2000, seed: int = 42, confidence: float = 0.95) -> dict[str, float]:
    y = np.asarray(y_true); a = np.asarray(score_a); b = np.asarray(score_b)
    if not (len(y) == len(a) == len(b)) or not len(y):
        raise ValueError("paired non-empty arrays are required")
    rng = np.random.default_rng(seed); differences = []
    for _ in range(iterations):
        index = rng.integers(0, len(y), len(y))
        try: differences.append(float(metric(y[index], a[index]) - metric(y[index], b[index])))
        except ValueError: continue
    if not differences:
        raise ValueError("metric was undefined for every bootstrap sample")
    values = np.asarray(differences); alpha = (1 - confidence) / 2
    return {"difference": float(metric(y, a) - metric(y, b)), "ci_low": float(np.quantile(values, alpha)), "ci_high": float(np.quantile(values, 1 - alpha)), "p_two_sided": float(2 * min(np.mean(values <= 0), np.mean(values >= 0))), "iterations_valid": int(len(values)), "seed": seed}


def paired_effect_size(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    differences = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    if len(differences) < 2 or differences.std(ddof=1) == 0:
        return 0.0
    return float(differences.mean() / differences.std(ddof=1))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float); order = np.argsort(values); adjusted = np.empty(len(values)); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index]); adjusted[index] = min(1.0, running)
    return adjusted.tolist()
