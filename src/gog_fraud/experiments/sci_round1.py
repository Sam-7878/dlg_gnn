"""Result contracts, checkpointing, and paper-table aggregation for round 1."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROUND1_REQUIRED_COLUMNS = (
    "run_id", "experiment_key", "config_hash", "dataset", "domain",
    "domain_group", "label_provenance", "model", "model_module", "model_class",
    "seed", "variant", "split_type", "roc_auc", "pr_auc",
    "oracle_best_f1", "oracle_best_threshold", "validation_f1",
    "validation_threshold", "f1_at_05", "topk_f1", "precision_at_k",
    "recall_at_k", "train_time_sec", "inference_time_sec", "peak_ram_mb",
    "peak_vram_mb", "num_nodes", "num_edges", "num_positive", "num_negative",
    "positive_ratio", "status", "error_type", "error_message", "traceback_path",
)


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def experiment_key(*, dataset: str, model: str, seed: int, variant: str,
                   split_strategy: str, config_hash: str) -> str:
    raw = "|".join((dataset, model, str(int(seed)), variant, split_strategy, config_hash))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_round1_record(record: Mapping[str, Any]) -> list[str]:
    errors = [f"missing column: {column}" for column in ROUND1_REQUIRED_COLUMNS if column not in record]
    if record.get("status") not in {"success", "failed", "oom", "timeout", "skipped", "unsupported"}:
        errors.append("invalid status")
    for metric in ("roc_auc", "pr_auc", "oracle_best_f1", "validation_f1", "f1_at_05", "topk_f1", "positive_ratio"):
        value = record.get(metric)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            if not 0.0 <= float(value) <= 1.0:
                errors.append(f"{metric} must be within [0, 1]")
    if all(record.get(name) is not None for name in ("num_nodes", "num_positive", "num_negative")):
        if int(record["num_positive"]) + int(record["num_negative"]) != int(record["num_nodes"]):
            errors.append("num_positive + num_negative must equal num_nodes")
    return errors


@dataclass
class ResultStore:
    path: Path
    rows: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def open(cls, path: str | Path) -> "ResultStore":
        target = Path(path)
        if target.is_file():
            return cls(target, pd.read_csv(target).to_dict(orient="records"))
        return cls(target)

    @property
    def completed_keys(self) -> set[str]:
        return {str(row["experiment_key"]) for row in self.rows if row.get("status") == "success"}

    def append(self, record: Mapping[str, Any]) -> None:
        errors = validate_round1_record(record)
        if errors:
            raise ValueError("; ".join(errors))
        self.rows.append(dict(record))
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self.rows)
        columns = list(ROUND1_REQUIRED_COLUMNS) + [column for column in frame.columns if column not in ROUND1_REQUIRED_COLUMNS]
        frame.reindex(columns=columns).to_csv(self.path, index=False)


def summarize_multiseed(frame: pd.DataFrame, *, metrics: tuple[str, ...] = (
    "roc_auc", "pr_auc", "oracle_best_f1", "validation_f1", "f1_at_05", "topk_f1",
)) -> pd.DataFrame:
    success = frame.loc[frame["status"].eq("success")].copy()
    records: list[dict[str, Any]] = []
    for (dataset, model), group in success.groupby(["dataset", "model"], sort=True):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            mean = float(values.mean())
            if values.size > 1:
                std = float(values.std(ddof=1)); half = 1.96 * std / np.sqrt(values.size)
            else:
                std = float("nan"); half = float("nan")
            records.append({
                "dataset": dataset, "model": model, "metric": metric,
                "mean": mean, "std": std, "median": float(np.median(values)),
                "min": float(values.min()), "max": float(values.max()),
                "ci95_low": mean - half, "ci95_high": mean + half,
                "n_seeds": int(values.size),
            })
    return pd.DataFrame.from_records(records)


def export_architecture_metadata(path: str | Path, *, model: Any, config: Mapping[str, Any],
                                 paper_name: str, variant: str) -> Path:
    """Export actual class identity plus resolved constructor/config values."""
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "paper_name": paper_name, "variant": variant,
        "model": {"module": model.__class__.__module__, "class": model.__class__.__qualname__},
        "resolved_config": dict(config),
    }
    for name in ("hid_dim", "num_layers", "l1_hops", "l1_epochs", "l1_hid_dim", "dropout", "weight", "backbone"):
        if hasattr(model, name):
            value = getattr(model, name)
            payload.setdefault("runtime_attributes", {})[name] = getattr(value, "__name__", value)
    target.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
    return target

