# src/gog_fraud/selection/thresholds.py
"""
Threshold optimization and selection policies for DLG-StreamMC selective routing.
Strict validation-only parameter selection ensuring test partition integrity.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .router import SelectiveRouter, TriageOutput


@dataclass(frozen=True)
class RoutingThresholds:
    tau_b: float
    tau_f: float
    tau_u: float
    version: str
    selected_on: str = "validation"
    metadata: Optional[Dict[str, Any]] = None

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), sort_keys=True, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RoutingThresholds":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def compute_variance_threshold(
    variances: Sequence[float],
    mode: str = "validation_quantile",
    param: float = 0.90,
    split: str = "validation",
) -> float:
    """
    Compute uncertainty threshold tau_u from validation partition.
    Modes:
      - 'absolute': Direct value of tau_u = param
      - 'validation_quantile': tau_u = quantile(variances, param) where param in [0, 1]
      - 'normalized': tau_u = mean(variances) + param * std(variances)
      - 'risk_controlled': upper bound percentile to isolate high variance tail
    """
    if split != "validation":
        raise ValueError(f"Variance thresholds must be computed on validation split only (got {split})")
    if not len(variances):
        raise ValueError("Variances sequence cannot be empty")

    arr = np.asarray(variances, dtype=float)

    if mode == "absolute":
        return float(param)
    elif mode in ("validation_quantile", "quantile", "risk_controlled"):
        if not (0.0 <= param <= 1.0):
            raise ValueError(f"Quantile param must be in [0, 1], got {param}")
        return float(np.quantile(arr, param))
    elif mode == "normalized":
        return float(arr.mean() + param * arr.std())
    else:
        raise ValueError(f"Unknown variance threshold mode: {mode}")


def optimize_compute_constrained(
    y_true: Sequence[int],
    scores: Sequence[float],
    uncertainties: Sequence[float],
    *,
    deep_budget: float,
    candidates: Iterable[Tuple[float, float, float]],
    split: str = "validation",
) -> RoutingThresholds:
    """
    Find optimal (tau_b, tau_f, tau_u) among candidates that satisfies deep_rate <= deep_budget on validation data.
    """
    if split != "validation":
        raise ValueError("threshold optimization is allowed on validation data only")
    if not (len(y_true) == len(scores) == len(uncertainties)) or not y_true:
        raise ValueError("non-empty arrays with equal lengths are required")

    best_key: Optional[Tuple[float, float, float, float, float]] = None
    best_thresholds: Optional[Tuple[float, float, float]] = None

    for tau_b, tau_f, tau_u in candidates:
        router = SelectiveRouter(tau_b=tau_b, tau_f=tau_f, tau_u=tau_u, threshold_version="candidate")
        decisions = [
            router.route(TriageOutput(score, unc, unc ** 0.5, 0.0, None, 1))
            for score, unc in zip(scores, uncertainties)
        ]
        deep_rate = sum(item.route == "deep_inspection" for item in decisions) / len(decisions)
        if deep_rate > deep_budget:
            continue
        false_negatives = sum(label == 1 and decision.route == "benign_direct" for label, decision in zip(y_true, decisions))
        positives = max(1, sum(int(label == 1) for label in y_true))
        objective = false_negatives / positives
        key = (objective, -deep_rate, tau_b, tau_f, tau_u)
        if best_key is None or key < best_key:
            best_key, best_thresholds = key, (tau_b, tau_f, tau_u)

    if best_thresholds is None:
        raise ValueError("no candidate satisfies the deep-route budget")

    tau_b, tau_f, tau_u = best_thresholds
    return RoutingThresholds(tau_b, tau_f, tau_u, f"validation_budget_{deep_budget:g}")


def grid_search_2d_thresholds(
    y_valid: Sequence[int],
    scores_valid: Sequence[float],
    uncertainties_valid: Sequence[float],
    tau_u: float,
    tau_b_grid: Sequence[float] = np.linspace(0.05, 0.45, 9),
    tau_f_grid: Sequence[float] = np.linspace(0.55, 0.95, 9),
    deep_budget: float = 0.50,
    split: str = "validation",
) -> RoutingThresholds:
    """
    2D Grid sweep of tau_b and tau_f on validation partition given fixed tau_u.
    Enforces tau_b < 0.5 < tau_f.
    """
    if split != "validation":
        raise ValueError("2D grid search is strictly restricted to validation data.")

    candidates = []
    for tb in tau_b_grid:
        for tf in tau_f_grid:
            if tb < 0.5 < tf:
                candidates.append((float(tb), float(tf), float(tau_u)))

    return optimize_compute_constrained(
        y_valid,
        scores_valid,
        uncertainties_valid,
        deep_budget=deep_budget,
        candidates=candidates,
        split=split,
    )
