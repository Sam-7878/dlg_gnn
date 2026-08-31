from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def reliability_rows(y_true, probabilities, *, bins: int = 10) -> list[dict[str, Any]]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 0.0, 1.0)
    if len(y) != len(p):
        raise ValueError("labels and probabilities must have equal length")
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(p, edges, right=False) - 1, bins - 1)
    rows = []
    for index in range(bins):
        mask = indices == index
        rows.append({
            "bin": index + 1, "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "count": int(mask.sum()), "mean_confidence": float(p[mask].mean()) if mask.any() else None,
            "empirical_rate": float(y[mask].mean()) if mask.any() else None,
        })
    return rows


def _ece(rows: list[dict[str, Any]], total: int) -> float:
    return float(sum(row["count"] / total * abs(row["mean_confidence"] - row["empirical_rate"]) for row in rows if row["count"])) if total else float("nan")


def _adaptive_ece(y: np.ndarray, p: np.ndarray, bins: int) -> float:
    groups = np.array_split(np.argsort(p), bins)
    return float(sum(len(group) / len(y) * abs(p[group].mean() - y[group].mean()) for group in groups if len(group))) if len(y) else float("nan")


def binary_calibration_metrics(y_true, probabilities) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 1e-12, 1 - 1e-12)
    if len(y) != len(p) or not len(y):
        raise ValueError("non-empty equal-length labels and probabilities are required")
    benign_ece = _ece(reliability_rows((y == 0).astype(int), 1.0 - p, bins=10), len(y))
    fraud_ece = _ece(reliability_rows((y == 1).astype(int), p, bins=10), len(y))
    return {
        "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "ece10": _ece(reliability_rows(y, p, bins=10), len(y)),
        "ece20": _ece(reliability_rows(y, p, bins=20), len(y)),
        "adaptive_ece": _adaptive_ece(y, p, 10),
        "classwise_ece": float(0.5 * (benign_ece + fraud_ece)),
        "fraud_class_ece": float(fraud_ece),
        "benign_class_ece": float(benign_ece),
    }


def fit_temperature(validation_labels, validation_logits) -> float:
    y = np.asarray(validation_labels, dtype=int).reshape(-1)
    logits = np.asarray(validation_logits, dtype=float).reshape(-1)
    temperatures = np.geomspace(0.05, 10.0, 500)
    losses = []
    for temperature in temperatures:
        probabilities = 1.0 / (1.0 + np.exp(-logits / temperature))
        losses.append(binary_calibration_metrics(y, probabilities)["nll"])
    return float(temperatures[int(np.argmin(losses))])


def write_reliability_csv(path: str | Path, y_true, probabilities, *, bins: int = 20) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    rows = reliability_rows(y_true, probabilities, bins=bins)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
