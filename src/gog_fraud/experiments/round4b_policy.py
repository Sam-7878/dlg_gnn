"""Round 4B algorithmic-support and complete-case benchmark policies."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

SUPPORTED_STATUSES = {"success"}
EXPECTED_UNSUPPORTED = {"unsupported_algorithmic"}
FAILURE_STATUSES = {
    "failed_numerical", "failed_cuda", "failed_oom_unexpected", "failed_data"
}


def classify_exact_runtime(seconds: float, *, prohibitive_hours: float = 24.0) -> str:
    if seconds < 0:
        raise ValueError("runtime must be non-negative")
    return "unsupported_algorithmic" if seconds > prohibitive_hours * 3600 else "supported_exact"


@dataclass(frozen=True)
class SupportCell:
    dataset: str
    model: str
    exact_backend_available: bool
    full_graph_feasible: bool
    reason_if_not: str | None
    primary_metric_available: bool
    status: str

    def __post_init__(self):
        allowed = SUPPORTED_STATUSES | EXPECTED_UNSUPPORTED | FAILURE_STATUSES | {"not_attempted"}
        if self.status not in allowed:
            raise ValueError(f"unknown support status: {self.status}")
        if self.status == "unsupported_algorithmic" and not self.reason_if_not:
            raise ValueError("algorithmic unsupported cells require an explicit reason")

    def to_dict(self):
        return asdict(self)


def complete_case_blocks(
    frame: pd.DataFrame,
    *,
    models: list[str],
    metric: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return matched dataset blocks; never impute unsupported/failure cells."""
    required = {"dataset", "model", "status", metric}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    selected = frame.loc[
        frame.model.isin(models) & frame.status.isin(SUPPORTED_STATUSES),
        ["dataset", "model", metric],
    ]
    pivot = selected.pivot_table(index="dataset", columns="model", values=metric, aggfunc="mean")
    pivot = pivot.reindex(columns=models).dropna(axis=0, how="any")
    return pivot, {
        "models": list(models),
        "datasets": list(pivot.index),
        "n_blocks": int(len(pivot)),
        "missing_policy": "complete_cases_only_no_rank_imputation",
    }
