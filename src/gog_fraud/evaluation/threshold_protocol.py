"""Leakage-explicit threshold selection and test evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score


def _arrays(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y.size == 0 or y.size != score.size:
        raise ValueError("non-empty y_true and y_score with equal length are required")
    valid = np.isfinite(y) & np.isfinite(score)
    y, score = y[valid], score[valid]
    if y.size == 0 or not np.isin(y, (0, 1)).all():
        raise ValueError("valid binary labels are required")
    return y, score


def best_f1_threshold(y_true: Any, y_score: Any) -> tuple[float, float]:
    y, score = _arrays(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y, score)
    if thresholds.size == 0:
        return float(f1_score(y, score >= 0.5, zero_division=0)), 0.5
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom > 0)
    index = int(np.nanargmax(f1))
    return float(f1[index]), float(thresholds[index])


def topk_predictions(y_score: Any, k: int) -> np.ndarray:
    score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if score.size == 0:
        raise ValueError("scores must be non-empty")
    k = max(0, min(int(k), score.size))
    pred = np.zeros(score.size, dtype=np.int64)
    if k:
        # Stable ordering makes ties reproducible and guarantees exactly k positives.
        order = np.argsort(-score, kind="mergesort")
        pred[order[:k]] = 1
    return pred


@dataclass(frozen=True)
class ThresholdEvaluation:
    oracle_best_f1: float
    oracle_best_threshold: float
    validation_f1: float
    validation_threshold: float
    f1_at_05: float
    precision_at_05: float
    recall_at_05: float
    topk_f1: float
    precision_at_k: float
    recall_at_k: float
    topk_k: int
    topk_prevalence_source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_threshold_protocol(
    validation_y: Any,
    validation_score: Any,
    test_y: Any,
    test_score: Any,
    *,
    topk_prevalence: float | None = None,
) -> ThresholdEvaluation:
    """Select a threshold only on validation labels, then evaluate on test.

    Oracle metrics remain available for retrospective analysis but are named as
    such. The default top-k budget is also derived from validation prevalence.
    """
    val_y, val_score = _arrays(validation_y, validation_score)
    y, score = _arrays(test_y, test_score)
    _, validation_threshold = best_f1_threshold(val_y, val_score)
    validation_pred = score >= validation_threshold
    oracle_f1, oracle_threshold = best_f1_threshold(y, score)
    fixed_pred = score >= 0.5

    if topk_prevalence is None:
        prevalence = float(val_y.mean())
        source = "validation_labels"
    else:
        prevalence = float(topk_prevalence)
        source = "configured"
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError("topk_prevalence must be within [0, 1]")
    k = int(round(prevalence * y.size))
    topk_pred = topk_predictions(score, k)

    return ThresholdEvaluation(
        oracle_best_f1=oracle_f1,
        oracle_best_threshold=oracle_threshold,
        validation_f1=float(f1_score(y, validation_pred, zero_division=0)),
        validation_threshold=validation_threshold,
        f1_at_05=float(f1_score(y, fixed_pred, zero_division=0)),
        precision_at_05=float(precision_score(y, fixed_pred, zero_division=0)),
        recall_at_05=float(recall_score(y, fixed_pred, zero_division=0)),
        topk_f1=float(f1_score(y, topk_pred, zero_division=0)),
        precision_at_k=float(precision_score(y, topk_pred, zero_division=0)),
        recall_at_k=float(recall_score(y, topk_pred, zero_division=0)),
        topk_k=k,
        topk_prevalence_source=source,
    )

