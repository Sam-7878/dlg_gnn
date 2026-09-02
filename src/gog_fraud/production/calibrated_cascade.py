"""Validation-only calibration and selection for production selective cascades.

The public functions in this module make the score contract explicit: every
model-facing score is a probability, calibration consumes log-odds, and fusion
is performed only in calibrated log-odds space.  Test labels are never accepted
by the selector API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def probability_to_logit(score: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1 or not np.isfinite(score).all():
        raise ValueError("score must be a finite one-dimensional probability array")
    if np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("probability score lies outside [0, 1]")
    clipped = np.clip(score, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def logit_to_probability(logit: np.ndarray) -> np.ndarray:
    value = np.asarray(logit, dtype=np.float64)
    return np.where(value >= 0, 1.0 / (1.0 + np.exp(-value)), np.exp(value) / (1.0 + np.exp(value)))


@dataclass(frozen=True)
class PlattLogOddsCalibrator:
    coefficient: float
    intercept: float
    epsilon: float = 1e-6

    @classmethod
    def fit(
        cls,
        score: np.ndarray,
        labels: np.ndarray,
        *,
        logistic_c: float = 1.0,
        epsilon: float = 1e-6,
        random_state: int = 0,
    ) -> "PlattLogOddsCalibrator":
        labels = np.asarray(labels, dtype=np.int64)
        if labels.ndim != 1 or len(labels) != len(score) or np.unique(labels).size != 2:
            raise ValueError("calibration requires aligned binary validation labels")
        feature = probability_to_logit(score, epsilon).reshape(-1, 1)
        model = LogisticRegression(C=logistic_c, solver="lbfgs", random_state=random_state)
        model.fit(feature, labels)
        return cls(float(model.coef_[0, 0]), float(model.intercept_[0]), epsilon)

    def transform(self, score: np.ndarray) -> np.ndarray:
        return logit_to_probability(self.coefficient * probability_to_logit(score, self.epsilon) + self.intercept)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def weighted_logit_fusion(
    fast_probability: np.ndarray,
    deep_probability: np.ndarray,
    fast_weight: float,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    if not 0.0 <= fast_weight <= 1.0:
        raise ValueError("fast_weight must lie in [0, 1]")
    fast = probability_to_logit(fast_probability, epsilon)
    deep = probability_to_logit(deep_probability, epsilon)
    if fast.shape != deep.shape:
        raise ValueError("fast and deep scores must have identical shapes")
    return logit_to_probability(fast_weight * fast + (1.0 - fast_weight) * deep)


def select_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    best = max(candidates, key=lambda threshold: (f1_score(labels, scores >= threshold, zero_division=0), -abs(float(threshold) - 0.5)))
    return float(best)


def ambiguity_cutoff(scores: np.ndarray, threshold: float, deep_budget: float) -> float:
    if not 0.0 <= deep_budget <= 1.0:
        raise ValueError("deep_budget must lie in [0, 1]")
    distance = np.abs(np.asarray(scores, dtype=np.float64) - threshold)
    return float(np.quantile(distance, deep_budget, method="higher"))


@dataclass(frozen=True)
class CascadeSelection:
    fast_threshold: float
    route_cutoff: float
    fast_weight: float
    final_threshold: float
    requested_deep_budget: float
    validation_deep_rate: float
    validation_f1: float


def apply_cascade(
    fast_probability: np.ndarray,
    deep_probability: np.ndarray,
    selection: CascadeSelection,
    *,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fused = weighted_logit_fusion(fast_probability, deep_probability, selection.fast_weight, epsilon=epsilon)
    routed = np.abs(np.asarray(fast_probability) - selection.fast_threshold) <= selection.route_cutoff
    final = np.where(routed, fused, fast_probability)
    return final, routed, fused


def select_cascade_on_validation(
    labels: np.ndarray,
    fast_probability: np.ndarray,
    deep_probability: np.ndarray,
    *,
    fast_weight_grid: Iterable[float],
    deep_budget_grid: Iterable[float],
    epsilon: float = 1e-6,
) -> CascadeSelection:
    """Select all operating parameters from validation data only.

    The final decision threshold is deliberately optimized after route/fusion
    construction.  Reusing the fast threshold here was the historical bug that
    produced the apparent tabular-cascade collapse.
    """
    labels = np.asarray(labels, dtype=np.int64)
    fast_probability = np.asarray(fast_probability, dtype=np.float64)
    deep_probability = np.asarray(deep_probability, dtype=np.float64)
    if labels.shape != fast_probability.shape or labels.shape != deep_probability.shape:
        raise ValueError("validation labels and scores must be aligned")
    fast_threshold = select_f1_threshold(labels, fast_probability)
    candidates: list[CascadeSelection] = []
    for budget in deep_budget_grid:
        cutoff = ambiguity_cutoff(fast_probability, fast_threshold, float(budget))
        routed = np.abs(fast_probability - fast_threshold) <= cutoff
        for weight in fast_weight_grid:
            fused = weighted_logit_fusion(fast_probability, deep_probability, float(weight), epsilon=epsilon)
            final = np.where(routed, fused, fast_probability)
            final_threshold = select_f1_threshold(labels, final)
            value = f1_score(labels, final >= final_threshold, zero_division=0)
            candidates.append(CascadeSelection(
                fast_threshold=fast_threshold,
                route_cutoff=cutoff,
                fast_weight=float(weight),
                final_threshold=final_threshold,
                requested_deep_budget=float(budget),
                validation_deep_rate=float(routed.mean()),
                validation_f1=float(value),
            ))
    return max(candidates, key=lambda item: (item.validation_f1, -item.validation_deep_rate, -abs(item.fast_weight - 0.5)))
