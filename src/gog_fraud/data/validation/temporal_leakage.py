from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class LeakageIssue:
    sample_id: str
    source: str
    source_max_time: int
    prediction_time: int


@dataclass
class LeakageReport:
    checked: int = 0
    issues: list[LeakageIssue] = field(default_factory=list)
    scaler_fit_end: int | None = None
    train_end: int | None = None
    entity_overlap: dict[str, int] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.issues and (
            self.scaler_fit_end is None or self.train_end is None or self.scaler_fit_end <= self.train_end
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["valid"] = self.valid
        return result


def _get(record: Any, name: str, default: Any = None) -> Any:
    return record.get(name, default) if isinstance(record, Mapping) else getattr(record, name, default)


def validate_temporal_integrity(
    records: Iterable[Any],
    *,
    scaler_fit_end: int | None = None,
    train_end: int | None = None,
    split_entities: Mapping[str, Iterable[str]] | None = None,
) -> LeakageReport:
    report = LeakageReport(scaler_fit_end=scaler_fit_end, train_end=train_end)
    for record in records:
        report.checked += 1
        sample_id = str(_get(record, "sample_id"))
        prediction_time = int(_get(record, "event_time"))
        for source, field_name in (("feature", "feature_source_max_time"), ("relation", "relation_source_max_time")):
            source_time = _get(record, field_name)
            if source_time is not None and int(source_time) > prediction_time:
                report.issues.append(LeakageIssue(sample_id, source, int(source_time), prediction_time))
    if scaler_fit_end is not None and train_end is not None and scaler_fit_end > train_end:
        report.issues.append(LeakageIssue("__scaler__", "scaler", scaler_fit_end, train_end))
    if split_entities:
        sets = {name: set(values) for name, values in split_entities.items()}
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            report.entity_overlap[f"{left}_{right}"] = len(sets.get(left, set()) & sets.get(right, set()))
    return report
