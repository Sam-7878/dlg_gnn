from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gog_fraud.data.sci_v2.audit import audit_dataset
from gog_fraud.data.sci_v2.builder import BuildOptions, build_dataset, row_multiset_hash


def _fixture(tmp_path: Path) -> BuildOptions:
    raw = tmp_path / "dataset/transactions/ethereum"; raw.mkdir(parents=True)
    pd.DataFrame({"block_number": [2, 1, 1], "from": ["b", "a", "a"], "to": ["c", "b", "b"],
                  "transaction_hash": ["z", "y", "x"], "value": [2, 1, 1], "timestamp": [20, 10, 10]}).to_csv(raw / "0xabc.csv", index=False)
    labels = tmp_path / "dataset/labels.csv"
    pd.DataFrame({"Chain": ["ethereum"], "Contract": ["0xabc"], "Category": [1]}).to_csv(labels, index=False)
    mapping = tmp_path / "dataset/global_graph"; mapping.mkdir()
    (mapping / "ethereum_contract_to_number_mapping.json").write_text('{"0xabc": 7}', encoding="utf-8")
    return BuildOptions(raw.parent, tmp_path / "legacy", tmp_path / "out", labels, mapping, ("ethereum",), None, True)


def test_multiset_digest_is_order_independent_and_duplicate_sensitive() -> None:
    frame = pd.DataFrame({"a": [1, 2, 2], "b": ["x", "y", "y"]})
    assert row_multiset_hash(frame, ["a", "b"]) == row_multiset_hash(frame.iloc[::-1], ["a", "b"])
    assert row_multiset_hash(frame, ["a", "b"]) != row_multiset_hash(frame.iloc[:2], ["a", "b"])


def test_build_preserves_rows_and_embeds_cutoff_provenance(tmp_path: Path) -> None:
    options = _fixture(tmp_path); summary = build_dataset(options)
    assert summary["chains"]["ethereum"]["manifest_complete"]
    manifest = json.loads((options.output_root / "manifests/ethereum.json").read_text())
    record = manifest["records"][0]
    assert record["row_multiset_hash_before"] == record["row_multiset_hash_after"]
    assert record["sort_keys"] == ["timestamp", "block_number", "transaction_hash", "original_row_index"]
    result = audit_dataset(options.output_root, chains=("ethereum",))
    assert result["status"] == "PASS"  # leakage is independent of legacy compatibility
    assert result["violations"] == 0


def test_audit_detects_future_edge(tmp_path: Path) -> None:
    options = _fixture(tmp_path); build_dataset(options)
    manifest = json.loads((options.output_root / "manifests/ethereum.json").read_text())
    graph_path = Path(manifest["records"][0]["graph_path"])
    import torch
    graph = torch.load(graph_path, weights_only=False); graph["edge_time"][0] = graph["cutoff_time"] + 1; torch.save(graph, graph_path)
    # Hash validation catches tampering before any scientific use.
    result = audit_dataset(options.output_root, chains=("ethereum",))
    assert any(v["check"] == "graph_hash" for v in result["violation_records"])
