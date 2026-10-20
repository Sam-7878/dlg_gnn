"""Build the timestamp-grounded GoG SCI main-track dataset.

The input is the immutable GoG SCI v2 derivative.  Each detection event is a
contract-local transaction graph whose cutoff is a recorded on-chain timestamp.
The three chains are pooled before one fixed 70/15/15 chronological split;
Polygon alone is intentionally not used because its latest holdout has no
positive support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


VERSION = "gog-scimain-v1.0"
CHAINS = ("ethereum", "bsc", "polygon")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _utc(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _load_records(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    manifest_hashes: dict[str, str] = {}
    for chain in CHAINS:
        path = source_root / "manifests" / f"{chain}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("manifest_complete") or payload.get("failures"):
            raise RuntimeError(f"incomplete upstream manifest: {path}")
        manifest_hashes[chain] = sha256_file(path)
        for record in payload["records"]:
            row = dict(record)
            row["chain_id"] = chain
            records.append(row)
    records.sort(key=lambda row: (int(row["cutoff_time"]), row["sample_id"]))
    return records, manifest_hashes


def _class_support(rows: list[dict[str, Any]]) -> dict[str, int]:
    positive = sum(int(row["label"]) for row in rows)
    return {"n_events": len(rows), "n_positive": positive, "n_negative": len(rows) - positive}


def _split(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    n = len(records)
    train_end = int(n * SPLIT_FRACTIONS[0])
    validation_end = train_end + int(n * SPLIT_FRACTIONS[1])
    groups = {
        "train": records[:train_end],
        "validation": records[train_end:validation_end],
        "test": records[validation_end:],
    }
    for name, rows in groups.items():
        support = _class_support(rows)
        if support["n_positive"] <= 0 or support["n_negative"] <= 0:
            raise RuntimeError(f"{name} lacks binary class support: {support}")
    if not (
        groups["train"][-1]["cutoff_time"] <= groups["validation"][0]["cutoff_time"]
        <= groups["validation"][-1]["cutoff_time"] <= groups["test"][0]["cutoff_time"]
    ):
        raise RuntimeError("chronological split ordering failed")
    return groups


def _cap_graph(graph: dict[str, Any], max_edges: int) -> tuple[dict[str, torch.Tensor], int]:
    edge_index = graph["edge_index"].long()
    edge_time = graph["edge_time"].long()
    cutoff = int(graph["cutoff_time"])
    if edge_time.numel() and int(edge_time.max()) > cutoff:
        raise RuntimeError(f"future edge in {graph['sample_id']}")
    if edge_time.numel() > max_edges:
        order = torch.argsort(edge_time, stable=True)[-max_edges:]
        edge_index = edge_index[:, order]
        edge_time = edge_time[order]
    nodes = torch.unique(edge_index.reshape(-1), sorted=True)
    if nodes.numel() == 0:
        nodes = torch.tensor([0], dtype=torch.long)
        edge_index = torch.empty((2, 0), dtype=torch.long)
    remap = torch.full((int(graph["num_nodes"]),), -1, dtype=torch.long)
    remap[nodes] = torch.arange(nodes.numel())
    edge_index = remap[edge_index]
    max_edge_time = int(edge_time.max()) if edge_time.numel() else -1
    return {"x": graph["x"][nodes].float(), "edge_index": edge_index}, max_edge_time


def build_dataset(source_root: Path, output_root: Path, max_edges: int = 128) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    records, manifest_hashes = _load_records(source_root)
    groups = _split(records)
    split_by_id = {
        row["sample_id"]: name for name, rows in groups.items() for row in rows
    }
    chain_index = {chain: index for index, chain in enumerate(CHAINS)}

    packed_graphs: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    future_edge_violations = 0
    graph_hash_failures = 0
    for index, record in enumerate(records):
        graph_path = Path(record["graph_path"])
        if not graph_path.is_file():
            raise FileNotFoundError(graph_path)
        if sha256_file(graph_path) != record["graph_sha256"]:
            graph_hash_failures += 1
            raise RuntimeError(f"graph hash mismatch: {record['sample_id']}")
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        capped, max_input_time = _cap_graph(graph, max_edges)
        if max_input_time > int(record["cutoff_time"]):
            future_edge_violations += 1
        packed_graphs.append({
            **capped,
            "event_id": record["sample_id"],
            "timestamp": int(record["cutoff_time"]),
            "label": int(record["label"]),
            "chain_index": chain_index[record["chain_id"]],
            "max_input_edge_timestamp": max_input_time,
        })
        metadata_rows.append({
            "event_id": record["sample_id"],
            "chain_id": record["chain_id"],
            "contract_id": record["contract_id"],
            "timestamp": int(record["cutoff_time"]),
            "timestamp_utc": _utc(int(record["cutoff_time"])),
            "label": int(record["label"]),
            "label_category": int(record["label_category"]),
            "split": split_by_id[record["sample_id"]],
            "num_nodes": int(record["num_nodes"]),
            "num_edges": int(record["num_edges"]),
            "max_input_edge_timestamp": max_input_time,
            "future_edge_count": 0,
            "graph_path": str(graph_path),
            "graph_sha256": record["graph_sha256"],
            "sorted_transaction_path": record["sorted_path"],
            "sorted_transaction_sha256": record["sorted_sha256"],
            "original_source_sha256": record["source_sha256"],
            "original_source_available": Path(record["source_path"]).is_file(),
        })
        if (index + 1) % 1000 == 0:
            print(f"packed {index + 1}/{len(records)}", flush=True)

    graph_path = output_root / "graph.pt"
    torch.save({
        "version": VERSION,
        "max_edges_per_event": max_edges,
        "node_feature_names": ["log1p_in_degree", "log1p_out_degree", "log1p_total_degree"],
        "chain_names": list(CHAINS),
        "graphs": packed_graphs,
    }, graph_path)
    transactions_path = output_root / "transactions.parquet"
    pd.DataFrame(metadata_rows).to_parquet(transactions_path, index=False)

    rng = random.Random(20260829)
    audit_rows = rng.sample([row for row in metadata_rows if row["split"] == "test"], 20)
    audit_path = output_root / "future_edge_audit.csv"
    pd.DataFrame([{
        "event_id": row["event_id"],
        "event_timestamp": row["timestamp"],
        "max_input_edge_timestamp": row["max_input_edge_timestamp"],
        "num_neighbors": min(row["num_edges"], max_edges),
        "future_edge_count": row["future_edge_count"],
    } for row in audit_rows]).to_csv(audit_path, index=False)

    split_payload: dict[str, Any] = {
        "protocol": "fixed_pooled_temporal_holdout_70_15_15",
        "selection_policy": "fractions frozen before model training; class support is an acceptance gate, not a metric-tuning criterion",
        "groups": {},
    }
    for name, rows in groups.items():
        support = _class_support(rows)
        split_payload["groups"][name] = {
            **support,
            "start_time": int(rows[0]["cutoff_time"]),
            "end_time": int(rows[-1]["cutoff_time"]),
            "start_time_utc": _utc(int(rows[0]["cutoff_time"])),
            "end_time_utc": _utc(int(rows[-1]["cutoff_time"])),
            "fraud_ratio": support["n_positive"] / support["n_events"],
            "event_ids_sha256": canonical_hash([row["sample_id"] for row in rows]),
        }
    split_payload["split_hash"] = canonical_hash(split_payload)
    (output_root / "split_manifest.json").write_text(
        json.dumps(split_payload, indent=2) + "\n", encoding="utf-8"
    )

    label_semantics = source_root / "labels" / "label_semantics.json"
    manifest = {
        "dataset_name": "GoG-SCIMain-v1",
        "dataset_version": VERSION,
        "paper_eligible": future_edge_violations == 0 and graph_hash_failures == 0,
        "split_type": "chronological_real",
        "timestamp_source": "recorded_transaction_timestamp",
        "timestamp_semantics": "latest recorded on-chain event in each expanding contract graph",
        "timezone": "UTC (Unix epoch seconds)",
        "graph_source": "gog_sci_v2_pooled",
        "chains": list(CHAINS),
        "context_policy": "excluded_from_main_detector_metrics",
        "gnn_input_policy": f"most recent {max_edges} historical edges at or before event cutoff",
        "label_provenance": "GoG upstream category; category 0 mapped to fraud per SCI v2 resolved semantics",
        "label_semantics_sha256": sha256_file(label_semantics),
        "upstream_manifest_sha256": manifest_hashes,
        "source_availability": "immutable sorted derivatives and graph snapshots available; original source paths absent locally",
        "license_availability": "upstream version/license metadata not present in local archive; author verification required before redistribution",
        "n_events": len(records),
        "future_edge_count": future_edge_violations,
        "graph_hash_failures": graph_hash_failures,
        "graph_sha256": sha256_file(graph_path),
        "transactions_sha256": sha256_file(transactions_path),
        "split_manifest_sha256": sha256_file(output_root / "split_manifest.json"),
        "future_edge_audit_sha256": sha256_file(audit_path),
        "split": split_payload["groups"],
    }
    (output_root / "real_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/d/_Work/_data/GoG_sci_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("data/benchmark/gog_scimain_v1"))
    parser.add_argument("--max-edges", type=int, default=128)
    args = parser.parse_args()
    manifest = build_dataset(args.source_root, args.output_root, args.max_edges)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
