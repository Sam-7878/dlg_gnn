"""Shared, audit-oriented utilities for the SCI-v3 final evidence pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)


SEEDS = (11, 22, 33, 44, 55)
CHAINS = ("ethereum", "bsc", "polygon")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def binary_metrics(labels: Iterable[int], scores: Iterable[float], threshold: float) -> dict[str, Any]:
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    pred = s >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = np.unique(y).size == 2
    clipped = np.clip(s, 1e-7, 1 - 1e-7)
    return {
        "roc_auc": float(roc_auc_score(y, s)) if both else None,
        "pr_auc": float(average_precision_score(y, s)) if y.sum() else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "fraud_recall": float(tp / (tp + fn)) if tp + fn else None,
        "fnr": float(fn / (tp + fn)) if tp + fn else None,
        "mcc": float(matthews_corrcoef(y, pred)) if both else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
        "n_test": int(y.size),
        "n_fraud": int(y.sum()),
        "n_benign": int((y == 0).sum()),
        "fraud_prevalence": float(y.mean()) if y.size else None,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "nll": float(log_loss(y, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y, clipped)),
    }


def select_f1_threshold(labels: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    candidates = np.unique(np.quantile(s, np.linspace(0, 1, min(201, max(2, s.size)))))
    values = [f1_score(y, s >= candidate, zero_division=0) for candidate in candidates]
    return float(candidates[int(np.argmax(values))])


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(value if isinstance(value, list) else value.get("records", []))
    return pd.read_csv(path)
