from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .builder import FEATURE_PROVENANCE, canonical_json_hash, row_multiset_hash, sha256_file


REQUIRED_RECORD_FIELDS = {
    "sample_id", "contract_id", "chain_id", "source_sha256", "sorted_sha256",
    "graph_sha256", "label", "label_category", "event_start", "event_end",
    "cutoff_time", "num_nodes", "num_edges", "feature_provenance_hash",
}


def audit_dataset(dataset_root: str | Path, *, chains: tuple[str, ...] = ("ethereum", "bsc", "polygon"),
                  strict: bool = False) -> dict[str, Any]:
    root = Path(dataset_root)
    violations: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    checked = 0
    seen: set[str] = set()
    for chain in chains:
        manifest_path = root / f"manifests/{chain}.json"
        if not manifest_path.exists():
            incomplete.append({"chain": chain, "check": "manifest", "reason": "missing"}); continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("manifest_complete"):
            incomplete.append({"chain": chain, "check": "manifest_complete", "reason": manifest.get("failures")})
        split_path = root / f"splits/{chain}_holdout_v2.json"
        normalizer_path = root / f"normalizers/{chain}/holdout/normalizer.json"
        relation_path = root / f"relations/{chain}/holdout/relation_state.json"
        if not all(p.exists() for p in (split_path, normalizer_path, relation_path)):
            incomplete.append({"chain": chain, "check": "fold_preprocessing", "reason": "missing artifact"})
        else:
            split = json.loads(split_path.read_text(encoding="utf-8")); normalizer = json.loads(normalizer_path.read_text(encoding="utf-8")); relation = json.loads(relation_path.read_text(encoding="utf-8"))
            train_ids = set(split["groups"]["train"]["sample_ids"])
            if set(normalizer["train_sample_ids"]) != train_ids or normalizer.get("fit_scope") != "train_only":
                violations.append({"chain": chain, "check": "normalizer_train_only"})
            if not set(relation["candidate_pool"]).issubset(train_ids) or relation.get("future_nodes_included") != 0 or relation.get("future_relations_included") != 0:
                violations.append({"chain": chain, "check": "relation_future_candidate"})
        chain_records = manifest.get("records", [])
        print(f"[{chain}] leakage audit start: {len(chain_records)} records", file=sys.stderr, flush=True)
        for record_index, record in enumerate(chain_records, start=1):
            checked += 1
            missing = sorted(REQUIRED_RECORD_FIELDS - record.keys())
            if missing:
                incomplete.append({"sample_id": record.get("sample_id"), "check": "provenance", "missing": missing}); continue
            sid = record["sample_id"]
            if sid in seen: violations.append({"sample_id": sid, "check": "duplicate_sample_id"})
            seen.add(sid)
            sorted_path, graph_path = Path(record["sorted_path"]), Path(record["graph_path"])
            if not sorted_path.exists() or not graph_path.exists():
                incomplete.append({"sample_id": sid, "check": "artifact_exists"}); continue
            source_path = Path(record["source_path"])
            if not source_path.exists():
                incomplete.append({"sample_id": sid, "check": "source_exists"}); continue
            if sha256_file(source_path) != record["source_sha256"]:
                violations.append({"sample_id": sid, "check": "source_hash"})
            if sha256_file(sorted_path) != record["sorted_sha256"]:
                violations.append({"sample_id": sid, "check": "sorted_hash"})
            if sha256_file(graph_path) != record["graph_sha256"]:
                violations.append({"sample_id": sid, "check": "graph_hash"})
            frame = pd.read_csv(sorted_path, dtype=str, keep_default_na=False, low_memory=False)
            original_cols = [c for c in frame.columns if c != "original_row_index"]
            if len(frame) != record["row_count_before"] or len(frame) != record["row_count_after"]:
                violations.append({"sample_id": sid, "check": "row_count"})
            if row_multiset_hash(frame, original_cols) != record["row_multiset_hash_after"]:
                violations.append({"sample_id": sid, "check": "row_multiset"})
            source_frame = pd.read_csv(source_path, dtype=str, keep_default_na=False, low_memory=False)
            source_frame.columns = [str(c).replace("\ufeff", "").strip() for c in source_frame.columns]
            if row_multiset_hash(source_frame, list(source_frame.columns)) != record["row_multiset_hash_before"]:
                violations.append({"sample_id": sid, "check": "source_row_multiset"})
            del source_frame
            times = pd.to_numeric(frame["timestamp"], errors="coerce")
            if times.isna().any() or not times.is_monotonic_increasing:
                violations.append({"sample_id": sid, "check": "sorted_event_monotonicity"})
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            edge_time = graph["edge_time"]
            edge_max = int(edge_time.max()) if edge_time.numel() else -1
            if edge_max > int(record["cutoff_time"]):
                violations.append({"sample_id": sid, "check": "graph_future_edge", "edge_max": edge_max})
            if record["feature_provenance_hash"] != canonical_json_hash(FEATURE_PROVENANCE):
                violations.append({"sample_id": sid, "check": "feature_provenance_hash"})
            if record_index % 100 == 0 or record_index == len(chain_records):
                print(f"[{chain}] audit {record_index}/{len(chain_records)} violations={len(violations)} incomplete={len(incomplete)}",
                      file=sys.stderr, flush=True)
    semantics_path = root / "labels/label_semantics.json"
    if not semantics_path.exists():
        incomplete.append({"check": "label_semantics", "reason": "missing"})
    elif json.loads(semantics_path.read_text(encoding="utf-8")).get("semantic_status") != "RESOLVED":
        incomplete.append({"check": "label_semantics", "reason": "upstream/local semantic conflict"})
    status = "FAIL" if violations else ("INCOMPLETE" if incomplete else "PASS")
    result = {"status": status, "records_checked": checked, "violations": len(violations),
              "incomplete_checks": len(incomplete), "violation_records": violations,
              "incomplete_records": incomplete, "paper_eligible": status == "PASS"}
    audit_dir = root / "audit"; audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "leakage_audit_all.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with (audit_dir / "leakage_violations.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("kind", "record")); writer.writeheader()
        for row in violations: writer.writerow({"kind": "violation", "record": json.dumps(row, sort_keys=True)})
        for row in incomplete: writer.writerow({"kind": "incomplete", "record": json.dumps(row, sort_keys=True)})
    if strict and status != "PASS":
        raise RuntimeError(f"SCI v2 leakage gate is {status}: {len(violations)} violations, {len(incomplete)} incomplete")
    return result
