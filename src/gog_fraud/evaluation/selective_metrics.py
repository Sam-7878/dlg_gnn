# src/gog_fraud/evaluation/selective_metrics.py
"""
Risk-coverage and selective classification evaluation metrics for DLG-StreamMC.
Computes coverage, selective risk, AURC, E-AURC, and fraud FNR curves.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class RiskCoveragePoint:
    coverage: float
    n_covered: int
    selective_risk: float
    fraud_fnr_at_coverage: float
    precision_at_coverage: float
    threshold_value: float


@dataclass
class SelectiveRiskSummary:
    aurc: float
    e_aurc: float  # Excess AURC over optimal risk profile
    coverage_levels: List[float]
    risks: List[float]
    fraud_fnrs: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_risk_coverage_curve(
    y_true: Sequence[int],
    scores: Sequence[float],
    uncertainties: Sequence[float],
    decision_threshold: float = 0.5,
    num_points: int = 100,
) -> Tuple[List[RiskCoveragePoint], SelectiveRiskSummary]:
    """
    Computes risk-coverage profile by ranking samples from lowest to highest uncertainty.
    Coverage c is the fraction of top confident samples evaluated.
    Selective risk = error rate among covered samples.
    AURC = Area Under Risk-Coverage Curve.
    E-AURC = AURC - Optimal AURC (where all errors are rejected first).
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    u = np.asarray(uncertainties, dtype=float)

    n = len(y)
    if n == 0 or not (n == len(s) == len(u)):
        raise ValueError(f"Invalid input lengths: {n}, {len(s)}, {len(u)}")

    preds = (s >= decision_threshold).astype(int)
    errors = (preds != y).astype(int)
    total_frauds = int(np.sum(y == 1))

    # Sort samples by confidence (lowest uncertainty first)
    sort_idx = np.argsort(u)
    y_sorted = y[sort_idx]
    preds_sorted = preds[sort_idx]
    errors_sorted = errors[sort_idx]

    # Precompute cumulative error and fraud counts
    cum_errors = np.cumsum(errors_sorted)
    cum_frauds = np.cumsum(y_sorted == 1)
    cum_pred_frauds = np.cumsum(preds_sorted == 1)
    cum_true_positives = np.cumsum((preds_sorted == 1) & (y_sorted == 1))

    coverages = np.linspace(1.0 / n, 1.0, min(num_points, n))
    points: List[RiskCoveragePoint] = []
    risks_list: List[float] = []
    cov_list: List[float] = []
    fnrs_list: List[float] = []

    for cov in coverages:
        k = max(1, int(round(cov * n)))
        err_k = cum_errors[k - 1]
        risk_k = float(err_k / k)

        # Fraud FNR among total population: missed frauds in covered set + all frauds in uncovered set
        tp_k = cum_true_positives[k - 1]
        fn_total = total_frauds - tp_k
        fnr_k = float(fn_total / total_frauds) if total_frauds > 0 else 0.0

        pred_f_k = cum_pred_frauds[k - 1]
        prec_k = float(tp_k / pred_f_k) if pred_f_k > 0 else 0.0

        u_thresh = float(u[sort_idx[k - 1]])

        pt = RiskCoveragePoint(
            coverage=float(k / n),
            n_covered=k,
            selective_risk=risk_k,
            fraud_fnr_at_coverage=fnr_k,
            precision_at_coverage=prec_k,
            threshold_value=u_thresh,
        )
        points.append(pt)
        risks_list.append(risk_k)
        cov_list.append(float(k / n))
        fnrs_list.append(fnr_k)

    # Compute AURC via trapezoidal rule
    aurc = float(np.trapezoid(risks_list, cov_list))

    # Optimal risk curve: sorts errors so all errors are at the tail (rejected first)
    total_errors = int(np.sum(errors))
    n_correct = n - total_errors
    opt_risks = []
    for cov in cov_list:
        k = max(1, int(round(cov * n)))
        if k <= n_correct:
            opt_risks.append(0.0)
        else:
            opt_risks.append(float((k - n_correct) / k))
    opt_aurc = float(np.trapezoid(opt_risks, cov_list))
    e_aurc = float(max(0.0, aurc - opt_aurc))

    summary = SelectiveRiskSummary(
        aurc=aurc,
        e_aurc=e_aurc,
        coverage_levels=cov_list,
        risks=risks_list,
        fraud_fnrs=fnrs_list,
    )

    return points, summary


def risk_coverage_to_dataframe(points: Sequence[RiskCoveragePoint]) -> pd.DataFrame:
    return pd.DataFrame([asdict(p) for p in points])
