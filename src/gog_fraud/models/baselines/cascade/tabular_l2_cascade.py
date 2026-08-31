"""Budget-matched selective cascade composition without test-set threshold tuning."""
from __future__ import annotations

import numpy as np


def ambiguity_cutoff(validation_scores: np.ndarray, threshold: float, deep_budget: float) -> float:
    if not 0.0 <= deep_budget <= 1.0:
        raise ValueError("deep_budget must be in [0, 1]")
    margin = np.abs(np.asarray(validation_scores, dtype=float) - threshold)
    return float(np.quantile(margin, deep_budget))


def apply_budgeted_cascade(
    fast_scores: np.ndarray,
    deep_scores: np.ndarray,
    threshold: float,
    margin_cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    fast = np.asarray(fast_scores, dtype=float)
    deep = np.asarray(deep_scores, dtype=float)
    if fast.shape != deep.shape:
        raise ValueError("fast and deep scores must be sample-aligned")
    route_deep = np.abs(fast - threshold) <= margin_cutoff
    return np.where(route_deep, deep, fast), route_deep
