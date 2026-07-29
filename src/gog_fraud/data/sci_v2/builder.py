from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

VERSION = "gog-sci-v2.0"
FEATURE_VERSION = "feature-v2.0"
MAPPING_VERSION = "mapping-v2.1-upstream-category0-fraud"
SORT_CANDIDATES = ("timestamp", "block_number", "transaction_index", "transaction_hash")
INVALID_ADDRESSES = {"", "nan", "none", "null"}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def row_multiset_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    """Order-independent, duplicate-sensitive digest over the original columns.

    pandas' stable uint64 row hashes are accumulated using two commutative
    moments. The final SHA-256 names the algorithm and binds the row count.
    """
    values = pd.util.hash_pandas_object(frame[columns], index=False, categorize=False).to_numpy(np.uint64)
    total = int(values.sum(dtype=np.uint64))
    squares = int(np.multiply(values, values, dtype=np.uint64).sum(dtype=np.uint64))
    xor = int(np.bitwise_xor.reduce(values, initial=np.uint64(0)))
    return canonical_json_hash({
        "algorithm": "pandas-hash-v1-u64-sum-square-xor",
        "rows": len(frame), "sum": total, "sum_square": squares, "xor": xor,
    })


def _read_labels(path: Path) -> dict[str, dict[str, tuple[int, int]]]:
    result: dict[str, dict[str, tuple[int, int]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            chain = str(row["Chain"]).strip().lower()
            contract = str(row["Contract"]).strip().lower()
            category = int(float(row["Category"]))
            # Verified against the upstream README and legacy embedded-label
            # aggregate: category 0 is fraud. Round 2's opposite assumption is
            # intentionally not carried into SCI v2.
            result.setdefault(chain, {})[contract] = (1 if category == 0 else 0, category)
    return result


def _stable_sort(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.copy()
    frame.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
    frame["original_row_index"] = np.arange(len(frame), dtype=np.int64)
    keys: list[str] = []
    internal_keys: list[str] = []
    for key in SORT_CANDIDATES:
        if key in frame.columns:
            keys.append(key)
            if key in {"timestamp", "block_number", "transaction_index"}:
                internal = f"__sort_{key}"
                frame[internal] = pd.to_numeric(frame[key], errors="raise").astype("int64")
                internal_keys.append(internal)
            else:
                internal_keys.append(key)
    if "timestamp" not in keys:
        raise ValueError("timestamp column is required")
    keys.append("original_row_index")
    internal_keys.append("original_row_index")
    result = frame.sort_values(internal_keys, kind="stable", ignore_index=True)
    result = result.drop(columns=[c for c in result.columns if c.startswith("__sort_")])
    return result, keys


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(encoded, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def _build_graph(frame: pd.DataFrame, *, chain: str, contract: str, source_path: Path,
                 source_sha: str, label: int, category: int) -> tuple[dict[str, Any], dict[str, Any]]:
    src = frame["from"].astype(str).str.strip().str.lower()
    dst = frame["to"].astype(str).str.strip().str.lower()
    valid = ~src.isin(INVALID_ADDRESSES) & ~dst.isin(INVALID_ADDRESSES)
    work = frame.loc[valid].copy()
    src = src.loc[valid]; dst = dst.loc[valid]
    if work.empty:
        raise ValueError("no valid transaction endpoints")
    nodes = sorted(set(src) | set(dst) | {contract})
    index = {node: i for i, node in enumerate(nodes)}
    src_idx = torch.tensor([index[v] for v in src], dtype=torch.long)
    dst_idx = torch.tensor([index[v] for v in dst], dtype=torch.long)
    time_values = pd.to_numeric(work["timestamp"], errors="raise").astype("int64")
    times = torch.tensor(time_values.to_numpy(np.int64), dtype=torch.long)
    values = pd.to_numeric(work["value"].astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0.0)
    edge_attr = torch.tensor(np.log1p(np.maximum(values.to_numpy(float), 0.0)), dtype=torch.float32).reshape(-1, 1)
    n = len(nodes)
    in_deg = torch.bincount(dst_idx, minlength=n).float()
    out_deg = torch.bincount(src_idx, minlength=n).float()
    x = torch.stack((torch.log1p(in_deg), torch.log1p(out_deg), torch.log1p(in_deg + out_deg)), dim=1)
    cutoff = int(time_values.max())
    start = int(time_values.min())
    sample_id = f"{chain}:{contract}:{cutoff}"
    attrs = {
        "sample_id": sample_id, "contract_id": contract, "chain_id": chain,
        "graph_version": VERSION, "source_file": str(source_path), "source_sha256": source_sha,
        "label": int(label), "label_category": int(category), "event_start": start,
        "event_end": cutoff, "cutoff_time": cutoff, "num_nodes": n,
        "num_edges": int(work.shape[0]), "window_type": "expanding",
        "window_start": start, "window_end": cutoff,
        "edge_cutoff_rule": "event_time <= cutoff_time",
        "node_sampling_rule": "all endpoints observed at or before cutoff",
        "paper_scope": "streaming_main", "feature_version": FEATURE_VERSION,
        "observed_num_nodes": len(set(src) | set(dst)),
        "transaction_value_sum": round(float(values.sum()), 6),
    }
    # A plain tensor dictionary is deliberately used as the immutable wire
    # format. Consumers may construct torch_geometric.data.Data(**graph) at
    # runtime, while artifact creation/audit remains independent of compiled
    # PyG extensions and their ABI.
    graph: dict[str, Any] = {
        "x": x, "edge_index": torch.stack((src_idx, dst_idx)),
        "edge_attr": edge_attr, "edge_time": times, "num_nodes": n,
        **attrs,
    }
    return graph, attrs


def _legacy_fingerprint(path: Path) -> tuple[tuple[int, int, float | None], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features", {})
    value_sum: float | None = None
    if isinstance(features, dict) and features and all(len(row) >= 5 for row in features.values()):
        # Upstream reference schema (raw degree/value aggregates).
        doubled_value = sum(float(row[3]) + float(row[4]) for row in features.values())
        value_sum = round(doubled_value / 2.0, 6)
    # Local legacy artifacts use a transformed list-of-lists schema. Its value
    # columns cannot be compared to raw totals, so mapping falls back to a
    # unique (edge_count, observed_node_count) shape and records that method.
    return (len(payload.get("edges", [])), len(features), value_sum), int(payload["label"])


def _resolve_legacy_mapping(records: list[dict[str, Any]], legacy_dir: Path) -> dict[str, Any]:
    by_exact: dict[tuple[int, int, float], list[dict[str, Any]]] = {}
    by_shape: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        exact = (record["num_edges"], record["observed_num_nodes"], record["transaction_value_sum"])
        by_exact.setdefault(exact, []).append(record)
        by_shape.setdefault(exact[:2], []).append(record)
    resolved: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    orientation = {"consistent": 0, "reversed": 0, "other": 0}
    for path in sorted(legacy_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        try:
            fingerprint, embedded = _legacy_fingerprint(path)
            candidates = by_exact.get(fingerprint, []) if fingerprint[2] is not None else []
            method = "exact_shape_value"
            if len(candidates) != 1:
                candidates = by_shape.get(fingerprint[:2], [])
                method = "unique_shape_fallback"
            if len(candidates) == 1:
                record = candidates[0]
                relation = "consistent" if embedded == record["label"] else ("reversed" if embedded == 1 - record["label"] else "other")
                orientation[relation] += 1
                resolved.append({"legacy_local_graph_id": int(path.stem), "contract_id": record["contract_id"],
                                 "sample_id": record["sample_id"], "embedded_label": embedded,
                                 "v2_binary_label": record["label"], "label_relation": relation,
                                 "mapping_method": method, "fingerprint": fingerprint})
            elif candidates:
                ambiguous.append({"legacy_local_graph_id": path.stem, "candidate_count": len(candidates),
                                  "candidate_contract_ids": [r["contract_id"] for r in candidates], "fingerprint": fingerprint})
            else:
                missing.append({"legacy_local_graph_id": path.stem, "fingerprint": fingerprint})
        except Exception as exc:
            missing.append({"legacy_local_graph_id": path.stem, "error": f"{type(exc).__name__}: {exc}"})
    total = len(resolved) + len(ambiguous) + len(missing)
    return {"status": "PASS" if total and len(resolved) == total else "INCOMPLETE",
            "legacy_graphs": total, "resolved": len(resolved), "ambiguous": len(ambiguous),
            "missing": len(missing), "label_orientation": orientation,
            "resolved_records": resolved, "ambiguous_records": ambiguous, "missing_records": missing}


def _build_fold_artifacts(out: Path, chain: str, records: list[dict[str, Any]]) -> None:
    ordered = sorted(records, key=lambda r: (r["event_end"], r["sample_id"]))
    n = len(ordered); train_end = max(1, int(n * 0.70)); valid_end = max(train_end + 1, int(n * 0.85)) if n > 2 else n
    groups = {"train": ordered[:train_end], "validation": ordered[train_end:valid_end], "test": ordered[valid_end:]}
    split = {
        "protocol": "fixed_temporal_holdout", "chain": chain, "dataset_version": VERSION,
        "groups": {name: {"sample_ids": [r["sample_id"] for r in rows],
                           "start_timestamp": min((r["event_end"] for r in rows), default=None),
                           "end_timestamp": max((r["event_end"] for r in rows), default=None)} for name, rows in groups.items()},
        "entity_overlap": {"train_validation": 0, "train_test": 0, "validation_test": 0},
    }
    split["split_hash"] = canonical_json_hash(split)
    _atomic_json(out / f"splits/{chain}_holdout_v2.json", split)
    feature_names = ["log1p_num_nodes", "log1p_num_edges", "log1p_transaction_value_sum"]
    matrix = np.asarray([[np.log1p(r["num_nodes"]), np.log1p(r["num_edges"]),
                          np.log1p(max(0.0, r["transaction_value_sum"]))] for r in groups["train"]], dtype=float)
    means = matrix.mean(axis=0) if len(matrix) else np.zeros(3)
    scales = matrix.std(axis=0) if len(matrix) else np.ones(3); scales[scales == 0] = 1.0
    normalizer = {
        "chain": chain, "fold": "holdout", "fit_scope": "train_only",
        "train_sample_ids": [r["sample_id"] for r in groups["train"]],
        "train_time_interval": [split["groups"]["train"]["start_timestamp"], split["groups"]["train"]["end_timestamp"]],
        "feature_names": feature_names, "means": means.tolist(), "scales": scales.tolist(),
        "code_version": FEATURE_VERSION,
    }
    normalizer["fit_hash"] = canonical_json_hash(normalizer)
    _atomic_json(out / f"normalizers/{chain}/holdout/normalizer.json", normalizer)
    cutoff = split["groups"]["train"]["end_timestamp"]
    pool = [r["sample_id"] for r in groups["train"] if cutoff is not None and r["event_end"] <= cutoff]
    relation = {
        "chain": chain, "fold": "holdout", "cutoff_time": cutoff,
        "candidate_pool": pool, "candidate_pool_hash": canonical_json_hash(pool),
        "candidate_count": len(pool), "relation_types": ["historical_candidate_pool"],
        "future_nodes_included": 0, "future_relations_included": 0,
    }
    relation["relation_state_hash"] = canonical_json_hash(relation)
    _atomic_json(out / f"relations/{chain}/holdout/relation_state.json", relation)


FEATURE_PROVENANCE = [
    {"feature_name": "in_degree", "source_columns": ["from", "to"], "aggregation": "count",
     "time_scope": "event_time <= cutoff_time", "fit_scope": "none", "version": FEATURE_VERSION},
    {"feature_name": "out_degree", "source_columns": ["from", "to"], "aggregation": "count",
     "time_scope": "event_time <= cutoff_time", "fit_scope": "none", "version": FEATURE_VERSION},
    {"feature_name": "transaction_value", "source_columns": ["value"], "aggregation": "log1p per event",
     "time_scope": "event_time <= cutoff_time", "fit_scope": "none", "version": FEATURE_VERSION},
]


@dataclass(frozen=True)
class BuildOptions:
    raw_root: Path
    legacy_root: Path
    output_root: Path
    labels_path: Path
    global_mapping_root: Path
    chains: tuple[str, ...] = ("ethereum", "bsc", "polygon")
    max_files: int | None = None
    strict: bool = False


def build_dataset(options: BuildOptions) -> dict[str, Any]:
    labels = _read_labels(options.labels_path)
    out = options.output_root
    for name in ("manifests", "labels", "features", "mappings", "splits", "normalizers", "relations", "audit"):
        (out / name).mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "features/feature_provenance.json", FEATURE_PROVENANCE)
    semantics = {
        "round2_local_rule": "category 0 = benign; every non-zero category = fraud",
        "v2_binary_rule": "category 0 = fraud (1); every non-zero category = benign (0)",
        "upstream_readme_statement": "Category 0 contains the most contracts: fraud",
        "semantic_status": "RESOLVED",
        "resolution": "SCI v2 follows the only available upstream semantic statement; legacy embedded label counts independently agree within missing-graph counts",
        "breaking_change": "Round 2 binary orientation was reversed and must not be reused",
        "upstream_version": "MISSING_UPSTREAM_VERSION_METADATA",
    }
    _atomic_json(out / "labels/label_semantics.json", semantics)
    summary: dict[str, Any] = {"dataset_version": VERSION, "chains": {}, "semantic_status": "RESOLVED"}
    for chain in options.chains:
        chain_labels = labels.get(chain, {})
        mapping_path = options.global_mapping_root / f"{chain}_contract_to_number_mapping.json"
        global_mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}
        paths = sorted((options.raw_root / chain).glob("*.csv"))
        if options.max_files is not None:
            paths = paths[:options.max_files]
        records: list[dict[str, Any]] = []
        mapping_rows: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        print(f"[{chain}] build start: {len(paths)} files", file=sys.stderr, flush=True)
        for file_index, path in enumerate(paths, start=1):
            contract = path.stem.lower()
            try:
                if contract not in chain_labels:
                    raise ValueError("missing label")
                original = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
                original.columns = [str(c).replace("\ufeff", "").strip() for c in original.columns]
                original_columns = list(original.columns)
                before_digest = row_multiset_hash(original, original_columns)
                sorted_frame, sort_keys = _stable_sort(original)
                after_digest = row_multiset_hash(sorted_frame, original_columns)
                if before_digest != after_digest or len(original) != len(sorted_frame):
                    raise RuntimeError("row multiset preservation failed")
                source_sha = sha256_file(path)
                sorted_path = out / "sorted_transactions" / chain / path.name
                _atomic_csv(sorted_frame, sorted_path)
                sorted_sha = sha256_file(sorted_path)
                label, category = chain_labels[contract]
                graph, graph_meta = _build_graph(sorted_frame, chain=chain, contract=contract,
                    source_path=path, source_sha=source_sha, label=label, category=category)
                graph_path = out / "graphs" / chain / f"{chain}__{contract}__{graph_meta['cutoff_time']}.pt"
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_graph = graph_path.with_suffix(".pt.tmp")
                torch.save(graph, tmp_graph); os.replace(tmp_graph, graph_path)
                timestamps = pd.to_numeric(sorted_frame["timestamp"], errors="raise").astype("int64")
                rec = {
                    **graph_meta, "source_path": str(path), "source_sha256": source_sha,
                    "sorted_path": str(sorted_path), "sorted_sha256": sorted_sha,
                    "graph_path": str(graph_path), "graph_sha256": sha256_file(graph_path),
                    "row_count_before": len(original), "row_count_after": len(sorted_frame),
                    "row_multiset_hash_before": before_digest, "row_multiset_hash_after": after_digest,
                    "was_reordered": not np.array_equal(sorted_frame["original_row_index"].to_numpy(), np.arange(len(original))),
                    "duplicate_rows": int(original.duplicated().sum()),
                    "duplicate_transactions": int(original["transaction_hash"].duplicated().sum()) if "transaction_hash" in original else None,
                    "sort_keys": sort_keys, "timestamp_min": int(timestamps.min()), "timestamp_max": int(timestamps.max()),
                    "feature_provenance_hash": canonical_json_hash(FEATURE_PROVENANCE),
                }
                records.append(rec)
                mapping_rows.append({
                    "sample_id": graph_meta["sample_id"], "chain_id": chain, "contract_id": contract,
                    "global_graph_id": global_mapping.get(contract), "legacy_local_graph_id": None,
                    "mapping_version": MAPPING_VERSION, "source_file": str(path), "graph_path": str(graph_path),
                })
                if file_index % 100 == 0 or file_index == len(paths):
                    print(f"[{chain}] {file_index}/{len(paths)} succeeded={len(records)} failed={len(failures)}",
                          file=sys.stderr, flush=True)
            except Exception as exc:
                failures.append({"source_file": str(path), "contract_id": contract,
                                 "error": f"{type(exc).__name__}: {exc}"})
                if options.strict:
                    raise
        manifest = {
            "dataset_version": VERSION, "chain": chain, "records": records, "failures": failures,
            "files_expected": len(paths), "files_succeeded": len(records), "files_failed": len(failures),
            "manifest_complete": len(records) == len(paths) and not failures,
            "raw_root": str(options.raw_root), "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        _build_fold_artifacts(out, chain, records)
        if options.max_files is None:
            legacy_audit = _resolve_legacy_mapping(records, options.legacy_root / chain / "graphs")
        else:
            legacy_audit = {"status": "NOT_RUN_SMOKE", "legacy_graphs": 0, "resolved": 0,
                            "ambiguous": 0, "missing": 0, "label_orientation": {},
                            "resolved_records": [], "ambiguous_records": [], "missing_records": []}
        _atomic_json(out / f"audit/{chain}_legacy_compatibility.json", legacy_audit)
        legacy_by_contract = {r["contract_id"]: r["legacy_local_graph_id"] for r in legacy_audit["resolved_records"]}
        for row in mapping_rows:
            row["legacy_local_graph_id"] = legacy_by_contract.get(row["contract_id"])
        manifest["legacy_mapping_status"] = legacy_audit["status"]
        manifest["legacy_mapping_resolved"] = legacy_audit["resolved"]
        _atomic_json(out / f"manifests/{chain}.json", manifest)
        _atomic_json(out / f"mappings/{chain}_raw_to_graph.json", mapping_rows)
        summary["chains"][chain] = {k: manifest[k] for k in ("files_expected", "files_succeeded", "files_failed", "manifest_complete")}
    _atomic_json(out / "manifests/dataset_summary.json", summary)
    card = """# GoG SCI Dataset v2\n\nLeakage-safe, immutable derivatives of the original transaction CSV corpus.\nOriginal files are never modified. Each graph is an expanding-window snapshot at\nits recorded cutoff and stores edge timestamps and source provenance.\n\nSCI v2 follows the upstream category-0-is-fraud statement. This is a breaking\ncorrection to the reversed Round 2 local assumption; see `labels/label_semantics.json`.\n"""
    card_path = out / "DATASET_CARD.md"
    if not card_path.exists():
        card_path.write_text(card, encoding="utf-8")
    return summary
