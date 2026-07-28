from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RollingOriginFold:
    fold: int
    train_ids: tuple[str, ...]
    valid_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    split_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get(record: Any, name: str) -> Any:
    return record.get(name) if isinstance(record, dict) else getattr(record, name)


def rolling_origin_splits(
    records: Iterable[Any],
    *,
    n_folds: int = 5,
    min_train_windows: int = 4,
    id_field: str = "sample_id",
    time_field: str = "event_time",
) -> list[RollingOriginFold]:
    """Build expanding-window train/valid/test folds from time buckets."""
    if n_folds < 1 or min_train_windows < 1:
        raise ValueError("n_folds and min_train_windows must be positive")
    ordered = sorted(records, key=lambda r: (int(_get(r, time_field)), str(_get(r, id_field))))
    window_count = min_train_windows + n_folds + 1
    if len(ordered) < window_count:
        raise ValueError(f"at least {window_count} records are required")
    boundaries = [round(i * len(ordered) / window_count) for i in range(window_count + 1)]
    windows = [ordered[boundaries[i]:boundaries[i + 1]] for i in range(window_count)]
    folds: list[RollingOriginFold] = []
    for index in range(n_folds):
        valid_window = min_train_windows + index
        test_window = valid_window + 1
        train = [record for window in windows[:valid_window] for record in window]
        valid, test = windows[valid_window], windows[test_window]
        ids = tuple(tuple(str(_get(r, id_field)) for r in group) for group in (train, valid, test))
        payload = {"fold": index + 1, "train": ids[0], "valid": ids[1], "test": ids[2]}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        folds.append(RollingOriginFold(index + 1, ids[0], ids[1], ids[2], digest))
    return folds
