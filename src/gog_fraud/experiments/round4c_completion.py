"""Evidence-preserving completion policy for the Round 4C production pilot.

Raw execution records are immutable observations.  A separate classification
ledger may turn a reproduced, explained resource failure into an operational
support decision without erasing the original failure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .round4c_policy import FAILURE_STATUSES, UNSUPPORTED_STATUSES, validate_status


def load_classification_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[
            "dataset", "model", "seed", "final_status", "evidence_mode",
            "restriction_reason", "evidence_path",
        ])
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("classifications", value) if isinstance(value, dict) else value
    frame = pd.DataFrame(rows)
    required = {"dataset", "model", "seed", "final_status", "evidence_mode",
                "restriction_reason", "evidence_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"classification ledger missing columns: {sorted(missing)}")
    for row in frame.itertuples():
        status = validate_status(row.final_status)
        if status not in UNSUPPORTED_STATUSES:
            raise ValueError(f"ledger may only classify unsupported cells: {status}")
        if row.evidence_mode not in {"measured", "policy"}:
            raise ValueError(f"invalid evidence_mode: {row.evidence_mode}")
    if frame.duplicated(["dataset", "model", "seed"]).any():
        raise ValueError("duplicate classification ledger cells")
    return frame


def account_cells(
    raw: pd.DataFrame,
    ledger: pd.DataFrame,
    expected: Iterable[tuple[str, str, int]],
) -> pd.DataFrame:
    """Return one immutable-observation + final-classification row per cell."""
    raw_index = {
        (row.dataset, row.model, int(row.seed)): row._asdict()
        for row in raw.itertuples(index=False)
    }
    ledger_index = {
        (row.dataset, row.model, int(row.seed)): row._asdict()
        for row in ledger.itertuples(index=False)
    }
    rows = []
    for dataset, model, seed in expected:
        key = (dataset, model, int(seed))
        observed = raw_index.get(key)
        classified = ledger_index.get(key)
        observed_status = observed.get("status") if observed else "not_attempted"
        if classified:
            final_status = validate_status(classified["final_status"])
            evidence_mode = classified["evidence_mode"]
            reason = classified["restriction_reason"]
            evidence_path = classified["evidence_path"]
        else:
            final_status = observed_status
            evidence_mode = "measured" if observed else "none"
            reason = observed.get("failure_message") if observed else "unattempted"
            evidence_path = observed.get("evidence_path", "") if observed else ""
        accounted = final_status == "success" or final_status in UNSUPPORTED_STATUSES
        rows.append({
            "dataset": dataset, "model": model, "seed": int(seed),
            "observed_status": observed_status, "final_status": final_status,
            "evidence_mode": evidence_mode, "accounted": bool(accounted),
            "restriction_reason": reason, "evidence_path": evidence_path,
        })
    return pd.DataFrame(rows)


def display_seed_status(row: pd.Series) -> str:
    status = row.final_status
    if status in UNSUPPORTED_STATUSES:
        suffix = "measured" if row.evidence_mode == "measured" else "by_policy"
        return f"{status}_{suffix}"
    return status


def frozen_support_matrix(accounting: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), group in accounting.groupby(["dataset", "model"], sort=True):
        seeds = {int(row.seed): row for _, row in group.iterrows()}
        seed42 = seeds.get(42)
        seed43 = seeds.get(43)
        statuses = [row.final_status for row in (seed42, seed43) if row is not None]
        primary = len(statuses) == 2 and all(status == "success" for status in statuses)
        accounted = len(statuses) == 2 and all(bool(row.accounted) for row in (seed42, seed43))
        unsupported = [row for row in (seed42, seed43) if row is not None and row.final_status in UNSUPPORTED_STATUSES]
        restriction_class = ";".join(sorted({row.final_status for row in unsupported}))
        restriction_reason = " | ".join(dict.fromkeys(
            str(row.restriction_reason) for row in unsupported if row.restriction_reason
        ))
        evidence_path = ";".join(dict.fromkeys(
            str(row.evidence_path) for row in (seed42, seed43)
            if row is not None and row.evidence_path
        ))
        rows.append({
            "dataset": dataset, "model": model,
            "seed42_status": display_seed_status(seed42) if seed42 is not None else "not_attempted",
            "seed43_status": display_seed_status(seed43) if seed43 is not None else "not_attempted",
            "exact_objective": True, "exact_message_semantics": True,
            "production_tested": bool(seed42 is not None and seed43 is not None
                                      and seed42.observed_status != "not_attempted"
                                      and seed43.observed_status != "not_attempted"),
            "accounted": bool(accounted), "primary_supported": bool(primary),
            "restriction_class": restriction_class,
            "restriction_reason": restriction_reason,
            "evidence_path": evidence_path,
        })
    return pd.DataFrame(rows)


def completion_decision(accounting: pd.DataFrame) -> tuple[str, list[str]]:
    reasons = []
    if not accounting.accounted.all():
        reasons.append(f"unaccounted cells: {int((~accounting.accounted).sum())}")
    unresolved = accounting.loc[
        accounting.final_status.isin(FAILURE_STATUSES)
        | accounting.final_status.isin({"not_attempted", "failed_unknown"})
    ]
    if not unresolved.empty:
        reasons.append(f"unresolved failures: {len(unresolved)}")
    if reasons:
        return "NOT_READY", reasons
    if accounting.final_status.isin(UNSUPPORTED_STATUSES).any():
        return "READY_WITH_RESTRICTIONS", ["all unsupported cells have frozen scientific classifications"]
    return "READY_FOR_FULL_RUN", []


def complete_case_views(
    support: pd.DataFrame,
    models: list[str],
    datasets: list[str],
    fraud_datasets: list[str],
) -> pd.DataFrame:
    supported = {
        (row.dataset, row.model) for row in support.itertuples()
        if bool(row.primary_supported)
    }
    all_model_datasets = [
        dataset for dataset in datasets
        if all((dataset, model) in supported for model in models)
    ]
    scalable_models = [model for model in models if model not in {"GADNR", "AnomalyDAE"}]
    scalable_datasets = [
        dataset for dataset in datasets
        if all((dataset, model) in supported for model in scalable_models)
    ]
    common_fraud_models = [
        model for model in models
        if all((dataset, model) in supported for dataset in fraud_datasets)
    ]
    rows = [
        {
            "view_name": "full_comparable_subset", "models": ";".join(models),
            "datasets": ";".join(all_model_datasets), "n_models": len(models),
            "n_datasets": len(all_model_datasets),
            "reason": "all eight historical models; not main inferential if dataset blocks are too few",
        },
        {
            "view_name": "scalable_detector_subset", "models": ";".join(scalable_models),
            "datasets": ";".join(scalable_datasets), "n_models": len(scalable_models),
            "n_datasets": len(scalable_datasets),
            "reason": "excludes production-limited GADNR and AnomalyDAE",
        },
        {
            "view_name": "fraud_oriented_comparable_subset", "models": ";".join(common_fraud_models),
            "datasets": ";".join(fraud_datasets), "n_models": len(common_fraud_models),
            "n_datasets": len(fraud_datasets),
            "reason": "largest model intersection across representative fraud-oriented datasets",
        },
    ]
    return pd.DataFrame(rows)
