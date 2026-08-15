"""Round 4C production-matrix resume, status, support, and forecast policy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FINAL_STATUSES = {
    "success", "unsupported_algorithmic", "unsupported_operational",
    "unsupported_resource_exact_implementation",
    "failed_numerical", "failed_cuda", "failed_oom", "failed_data", "failed_other",
    "failed_unknown",
}
UNSUPPORTED_STATUSES = {
    "unsupported_algorithmic", "unsupported_operational",
    "unsupported_resource_exact_implementation",
}
FAILURE_STATUSES = FINAL_STATUSES - {"success"} - UNSUPPORTED_STATUSES


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cell_key(dataset: str, model: str, seed: int, config_hash: str, backend_hash: str) -> str:
    return canonical_hash({
        "dataset": dataset, "model": model, "seed": int(seed),
        "config_hash": config_hash, "backend_hash": backend_hash,
    })[:24]


def should_skip(path: Path, *, resume: bool) -> bool:
    if not resume or not path.exists():
        return False
    status = json.loads(path.read_text(encoding="utf-8")).get("status")
    return status == "success" or status in UNSUPPORTED_STATUSES


def classify_timeout(model: str) -> str:
    return "unsupported_operational" if model == "AnomalyDAE" else "failed_other"


def validate_status(status: str) -> str:
    if status not in FINAL_STATUSES:
        raise ValueError(f"invalid production status: {status}")
    return status


def final_support_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "model", "seed", "status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    rows = []
    for (dataset, model), group in frame.groupby(["dataset", "model"], sort=True):
        statuses = {int(row.seed): validate_status(row.status) for row in group.itertuples()}
        seed42, seed43 = statuses.get(42, "not_attempted"), statuses.get(43, "not_attempted")
        production_tested = 42 in statuses and 43 in statuses
        primary_supported = production_tested and seed42 == seed43 == "success"
        unsupported = {seed42, seed43}.intersection(UNSUPPORTED_STATUSES)
        restriction = None
        if unsupported:
            restriction = ";".join(sorted(unsupported))
        elif not primary_supported:
            restriction = "unexpected_failure_or_incomplete"
        rows.append({
            "dataset": dataset, "model": model, "exact_backend": True,
            "production_tested": production_tested, "seed42_status": seed42,
            "seed43_status": seed43, "primary_supported": primary_supported,
            "restriction": restriction,
        })
    return pd.DataFrame(rows)


def runtime_forecast(frame: pd.DataFrame, *, round5_seeds: int = 5) -> dict[str, object]:
    success = frame.loc[frame.status.eq("success")].copy()
    if success.empty:
        raise ValueError("successful production runs are required")
    pair_counts = success.groupby(["dataset", "model"]).seed.nunique()
    if (pair_counts < 2).any():
        raise ValueError("two measured seeds are required for every supported cell")
    grouped = success.groupby(["dataset", "model"]).total_wall_sec
    per_cell = grouped.agg(["min", "median", "max"]).reset_index()
    result: dict[str, object] = {
        "method": "sum per-model-dataset two-seed min/median/max, each expanded to five seeds",
        "round5_seed_count": int(round5_seeds),
        "optimistic_sec": float(per_cell["min"].sum() * round5_seeds),
        "median_sec": float(per_cell["median"].sum() * round5_seeds),
        "pessimistic_sec": float(per_cell["max"].sum() * round5_seeds),
        "measured_supported_cells": int(len(per_cell)),
    }
    for name, selected in (
        ("quadratic_anomalydae", per_cell.model.eq("AnomalyDAE")),
        ("normal_detectors", ~per_cell.model.eq("AnomalyDAE")),
    ):
        part = per_cell.loc[selected]
        result[name] = {
            "optimistic_sec": float(part["min"].sum() * round5_seeds),
            "median_sec": float(part["median"].sum() * round5_seeds),
            "pessimistic_sec": float(part["max"].sum() * round5_seeds),
        }
    return result


def readiness_decision(frame: pd.DataFrame, expected: Iterable[tuple[str, str, int]]) -> tuple[str, list[str]]:
    expected_set = set(expected)
    attempted = {(r.dataset, r.model, int(r.seed)) for r in frame.itertuples()}
    reasons: list[str] = []
    if attempted != expected_set:
        reasons.append(f"production matrix incomplete: {len(attempted)}/{len(expected_set)}")
    failures = frame.loc[frame.status.isin(FAILURE_STATUSES)]
    if not failures.empty:
        reasons.append(f"unexpected supported-cell failures: {len(failures)}")
    if reasons:
        return "NOT_READY", reasons
    if frame.status.isin(UNSUPPORTED_STATUSES).any():
        return "READY_WITH_RESTRICTIONS", ["objectively recorded unsupported cells excluded"]
    return "READY_FOR_FULL_RUN", []
