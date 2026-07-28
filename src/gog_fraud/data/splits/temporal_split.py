from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class TemporalSplit:
    train_ids: tuple[str, ...]
    valid_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    train_end: int | None
    valid_end: int | None
    split_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name)


def temporal_split(
    records: Iterable[Any],
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    *,
    id_field: str = "sample_id",
    time_field: str = "event_time",
) -> TemporalSplit:
    """Split records without shuffling, using a stable chronological tie-break."""
    if not (0.0 < train_ratio < 1.0 and 0.0 <= valid_ratio < 1.0):
        raise ValueError("train_ratio and valid_ratio must be within [0, 1]")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1")
    ordered = sorted(
        records,
        key=lambda r: (int(_field(r, time_field)), str(_field(r, id_field))),
    )
    if len(ordered) < 3:
        raise ValueError("at least three records are required for a temporal split")
    n = len(ordered)
    train_end_idx = max(1, min(int(n * train_ratio), n - 2))
    valid_end_idx = max(train_end_idx + 1, min(int(n * (train_ratio + valid_ratio)), n - 1))
    groups = (ordered[:train_end_idx], ordered[train_end_idx:valid_end_idx], ordered[valid_end_idx:])
    ids = tuple(tuple(str(_field(r, id_field)) for r in group) for group in groups)
    boundaries = (
        int(_field(groups[0][-1], time_field)) if groups[0] else None,
        int(_field(groups[1][-1], time_field)) if groups[1] else None,
    )
    payload = {"train": ids[0], "valid": ids[1], "test": ids[2], "boundaries": boundaries}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return TemporalSplit(ids[0], ids[1], ids[2], boundaries[0], boundaries[1], digest)
