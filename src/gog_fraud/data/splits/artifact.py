from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .rolling_origin import rolling_origin_splits
from .temporal_split import temporal_split

GENERATOR_VERSION = "round2-split-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    """Write once; permit an identical regeneration without changing bytes."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        candidate = dict(payload)
        if "generated_at" in existing:
            candidate["generated_at"] = existing["generated_at"]
        if existing != candidate:
            raise RuntimeError(f"immutable split artifact differs: {path}")
        return
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_binary_labels(path: str | Path, chain: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    with Path(path).open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("Chain", "")).lower() != chain.lower():
                continue
            contract = str(row.get("Contract", "")).strip().lower()
            category = str(row.get("Category", "")).strip().lower()
            if contract:
                labels[contract] = 0 if category in {"0", "0.0", "benign", "normal"} else 1
    return labels


def scan_contract_records(transaction_root: str | Path, labels_path: str | Path, chain: str, *, max_files: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    chain_root = Path(transaction_root).resolve() / chain
    labels = load_binary_labels(labels_path, chain)
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    paths = sorted(chain_root.glob("*.csv"))
    if max_files is not None:
        paths = paths[:max_files]
    def scan_one(path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
        contract = path.stem.lower()
        try:
            event_time: int | None = None
            previous_time: int | None = None
            timestamp_count = 0
            monotonic = True
            chunks = pd.read_csv(
                path,
                usecols=lambda value: str(value).lower().replace("_", "")
                in {"timestamp", "blocktimestamp"},
                chunksize=200_000,
            )
            for frame in chunks:
                if frame.empty or not len(frame.columns):
                    continue
                series = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna().astype("int64")
                if series.empty:
                    continue
                first_time = int(series.iloc[0])
                if previous_time is not None and first_time < previous_time:
                    monotonic = False
                monotonic = monotonic and bool(series.is_monotonic_increasing)
                previous_time = int(series.iloc[-1])
                chunk_max = int(series.max())
                event_time = chunk_max if event_time is None else max(event_time, chunk_max)
                timestamp_count += len(series)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            return None, {"sample_id": contract, "reason": f"read_error:{type(exc).__name__}"}
        if contract not in labels:
            return None, {"sample_id": contract, "reason": "missing_label"}
        if event_time is None:
            return None, {"sample_id": contract, "reason": "missing_timestamp"}
        return ({
                "sample_id": contract, "event_time": event_time, "label": labels[contract],
                "feature_source_max_time": event_time, "relation_source_max_time": event_time,
                "transaction_count": timestamp_count, "timestamps_monotonic": monotonic,
                "source_file": path.name,
            }, None)

    # Four workers keep peak memory bounded while overlapping the many small
    # DrvFS file opens. executor.map preserves sorted input order, so split
    # artifacts remain deterministic.
    with ThreadPoolExecutor(max_workers=4) as executor:
        for record, exclusion in executor.map(scan_one, paths):
            if record is not None:
                records.append(record)
            if exclusion is not None:
                exclusions.append(exclusion)
    return records, exclusions


def _group_summary(ids: tuple[str, ...], by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [by_id[sample_id] for sample_id in ids]
    labels = Counter(row["label"] for row in rows)
    times = [row["event_time"] for row in rows]
    return {
        "sample_ids": list(ids), "sample_count": len(rows), "class_count": {"benign": labels[0], "fraud": labels[1]},
        "start_timestamp": min(times, default=None), "end_timestamp": max(times, default=None),
    }


def build_split_artifacts(
    *, transaction_root: str | Path, labels_path: str | Path, chain: str,
    source_manifest: str | Path, output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    records, exclusions = scan_contract_records(transaction_root, labels_path, chain)
    if len(records) < 10:
        raise ValueError(f"insufficient timestamped samples for {chain}: {len(records)}")
    fixed = temporal_split(records)
    rolling = rolling_origin_splits(records, n_folds=5)
    by_id = {row["sample_id"]: row for row in records}
    common = {
        "chain": chain, "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(Path(source_manifest)), "source_manifest_hash": sha256_file(source_manifest),
        "entity_definition": "contract_address", "threshold_source": "validation",
    }
    fixed_payload = common | {
        "protocol": "fixed_temporal_holdout", "split_hash": fixed.split_hash,
        "sample_records": records,
        "train": _group_summary(fixed.train_ids, by_id), "validation": _group_summary(fixed.valid_ids, by_id),
        "test": _group_summary(fixed.test_ids, by_id),
        "entity_overlap": {"train_validation": 0, "train_test": 0, "validation_test": 0},
    }
    rolling_payload = common | {
        "protocol": "rolling_origin_5", "folds": [
            {"fold": fold.fold, "split_hash": fold.split_hash, "train": _group_summary(fold.train_ids, by_id),
             "validation": _group_summary(fold.valid_ids, by_id), "test": _group_summary(fold.test_ids, by_id),
             "entity_overlap": {"train_validation": 0, "train_test": 0, "validation_test": 0}}
            for fold in rolling
        ],
    }
    raw_order_violation_ids = [
        row["sample_id"] for row in records if not row["timestamps_monotonic"]
    ]
    raw_order_violations = len(raw_order_violation_ids)
    audit_payload = common | {
        "records_checked": len(records), "violations": raw_order_violations,
        "raw_order_violation_sample_ids": raw_order_violation_ids,
        "raw_event_time_audit_definition": "timestamps within each raw contract CSV must be monotonically non-decreasing",
        "raw_event_time_audit": "PASS" if raw_order_violations == 0 else "FAIL", "processed_feature_provenance_audit": "NOT_OBSERVABLE",
        "normalizer_fit_interval_audit": "NOT_RUN", "relation_construction_interval_audit": "NOT_RUN",
        "knn_candidate_pool_audit": "NOT_RUN", "future_lifetime_feature_audit": "NOT_RUN",
        "status": "INCOMPLETE",
        "reason": "Legacy processed graph JSON does not contain source timestamps or normalizer/relation/KNN provenance; raw event-time ordering alone cannot establish sample-level leakage freedom.",
        "exclusions": exclusions,
    }
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    fixed_path = target / f"{chain}_holdout_v1.json"
    rolling_path = target / f"{chain}_rolling5_v1.json"
    audit_path = target / f"{chain}_leakage_audit_v1.json"
    for path, payload in ((fixed_path, fixed_payload), (rolling_path, rolling_payload), (audit_path, audit_payload)):
        _write_immutable_json(path, payload)
    return fixed_path, rolling_path, audit_path


def build_pooled_split_artifacts(*, split_dir: str | Path, chains: tuple[str, ...] = ("ethereum", "bsc", "polygon")) -> tuple[Path, Path, Path]:
    target = Path(split_dir)
    records: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    for chain in chains:
        path = target / f"{chain}_holdout_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources[chain] = sha256_file(path)
        for raw in payload.get("sample_records", []):
            row = dict(raw); row["sample_id"] = f"{chain}:{row['sample_id']}"; row["chain"] = chain
            records.append(row)
    fixed = temporal_split(records)
    rolling = rolling_origin_splits(records, n_folds=5)
    by_id = {row["sample_id"]: row for row in records}
    common = {
        "chain": "pooled", "chains": list(chains), "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(), "source_split_hashes": sources,
        "entity_definition": "chain:contract_address", "threshold_source": "validation",
    }
    fixed_payload = common | {
        "protocol": "fixed_temporal_holdout", "split_hash": fixed.split_hash, "sample_records": records,
        "train": _group_summary(fixed.train_ids, by_id), "validation": _group_summary(fixed.valid_ids, by_id),
        "test": _group_summary(fixed.test_ids, by_id),
        "entity_overlap": {"train_validation": 0, "train_test": 0, "validation_test": 0},
    }
    rolling_payload = common | {"protocol": "rolling_origin_5", "folds": [
        {"fold": fold.fold, "split_hash": fold.split_hash, "train": _group_summary(fold.train_ids, by_id),
         "validation": _group_summary(fold.valid_ids, by_id), "test": _group_summary(fold.test_ids, by_id),
         "entity_overlap": {"train_validation": 0, "train_test": 0, "validation_test": 0}}
        for fold in rolling
    ]}
    raw_order_violation_ids = [
        row["sample_id"] for row in records if not row.get("timestamps_monotonic", False)
    ]
    raw_order_violations = len(raw_order_violation_ids)
    audit_payload = common | {
        "records_checked": len(records), "violations": raw_order_violations,
        "raw_order_violation_sample_ids": raw_order_violation_ids,
        "raw_event_time_audit_definition": "timestamps within each raw contract CSV must be monotonically non-decreasing",
        "raw_event_time_audit": "PASS" if raw_order_violations == 0 else "FAIL",
        "processed_feature_provenance_audit": "NOT_OBSERVABLE", "status": "INCOMPLETE",
        "reason": "Pooled audit inherits missing processed feature provenance from chain audits.",
    }
    outputs = (target / "pooled_holdout_v1.json", target / "pooled_rolling5_v1.json", target / "pooled_leakage_audit_v1.json")
    for path, payload in zip(outputs, (fixed_payload, rolling_payload, audit_payload)):
        _write_immutable_json(path, payload)
    return outputs
