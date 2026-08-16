"""Frozen support and raw-result integrity policy for the final SCI benchmark."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


SUPPORT_STATUSES = {
    "supported",
    "unsupported_operational",
    "unsupported_resource_exact_implementation",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_support_matrix(frame: pd.DataFrame, datasets: list[str], models: list[str]) -> None:
    required = {"dataset", "model", "support_status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"support matrix missing columns: {sorted(missing)}")
    if frame.duplicated(["dataset", "model"]).any():
        raise ValueError("duplicate model-dataset support rows")
    expected = {(dataset, model) for dataset in datasets for model in models}
    observed = set(map(tuple, frame[["dataset", "model"]].itertuples(index=False, name=None)))
    if observed != expected:
        raise ValueError(f"support matrix coverage mismatch: missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    unknown = set(frame.support_status).difference(SUPPORT_STATUSES)
    if unknown:
        raise ValueError(f"unknown support statuses: {sorted(unknown)}")


def supported_run_count(frame: pd.DataFrame, seeds: list[int]) -> int:
    return int(frame.support_status.eq("supported").sum() * len(seeds))


def validate_final_raw(raw: pd.DataFrame, support: pd.DataFrame, seeds: list[int]) -> None:
    required = {"dataset", "model", "seed", "status"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"raw freeze missing columns: {sorted(missing)}")
    if raw.duplicated(["dataset", "model", "seed"]).any():
        raise ValueError("duplicate dataset/model/seed result")
    if not raw.status.eq("success").all():
        raise ValueError("benchmark_raw may contain successful performance rows only")
    expected_seeds = set(map(int, seeds))
    for row in support.itertuples(index=False):
        selected = raw.loc[raw.dataset.eq(row.dataset) & raw.model.eq(row.model)]
        observed = set(selected.seed.astype(int))
        if row.support_status == "supported" and observed != expected_seeds:
            raise ValueError(f"{row.dataset}/{row.model} does not have exactly five successful seeds")
        if row.support_status != "supported" and not selected.empty:
            raise ValueError(f"unsupported pair has performance rows: {row.dataset}/{row.model}")


def seed_first_summary(raw: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """One row per dataset/model with seed-level descriptive statistics."""
    from scipy.stats import t

    records: list[dict] = []
    for (dataset, model), group in raw.groupby(["dataset", "model"], sort=True):
        record: dict[str, object] = {"dataset": dataset, "model": model, "n_seeds": len(group)}
        for metric in metrics:
            values = group[metric].astype(float)
            mean, std = float(values.mean()), float(values.std(ddof=1))
            half = float(t.ppf(0.975, len(values)-1) * std / len(values) ** 0.5)
            record.update({
                f"{metric}_mean": mean, f"{metric}_std": std,
                f"{metric}_median": float(values.median()),
                f"{metric}_min": float(values.min()), f"{metric}_max": float(values.max()),
                f"{metric}_ci95_low": mean-half, f"{metric}_ci95_high": mean+half,
            })
        records.append(record)
    return pd.DataFrame(records)


def complete_case_views(support: pd.DataFrame, fraud_datasets: list[str]) -> pd.DataFrame:
    """Derive maximal transparent comparison views from the final ten-dataset matrix."""
    supported = support.assign(value=support.support_status.eq("supported")).pivot(
        index="dataset", columns="model", values="value"
    )
    all_models = list(supported.columns)
    broad_datasets = list(supported.index[supported.all(axis=1)])
    scalable_models = [model for model in all_models if int(supported[model].sum()) == len(supported.index)]
    # Largest common model intersection across fraud datasets.
    fraud = supported.loc[[name for name in fraud_datasets if name in supported.index]]
    fraud_models = list(fraud.columns[fraud.all(axis=0)])
    rows = [
        {"view_name":"broad_complete_case", "models":";".join(all_models),
         "datasets":";".join(broad_datasets), "n_models":len(all_models), "n_datasets":len(broad_datasets),
         "reason":"all eight historical models on common supported datasets"},
        {"view_name":"scalable_detector", "models":";".join(scalable_models),
         "datasets":";".join(list(supported.index)), "n_models":len(scalable_models), "n_datasets":len(supported),
         "reason":"models supported on every final dataset"},
        {"view_name":"fraud_oriented", "models":";".join(fraud_models),
         "datasets":";".join(list(fraud.index)), "n_models":len(fraud_models), "n_datasets":len(fraud),
         "reason":"largest common model subset on frozen fraud-oriented datasets"},
    ]
    return pd.DataFrame(rows)
