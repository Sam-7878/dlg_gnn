from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from gog_fraud.data.io.dataset_manifest import build_dataset_manifest


def _write_transactions(path: Path, value: str = "a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "timestamp", "from", "to", "value"])
        writer.writeheader()
        writer.writerow({"sample_id": "s1", "timestamp": 10, "from": "a", "to": "b", "value": value})


def test_hash_cache_is_invalidated_by_file_metadata(tmp_path: Path) -> None:
    source = tmp_path / "ethereum" / "transactions.csv"
    index = tmp_path / "hash-index.json"
    _write_transactions(source, "a")
    first = build_dataset_manifest(tmp_path, chain="ethereum", hash_index_path=index)
    first_digest = first.file_hashes["ethereum/transactions.csv"]
    _write_transactions(source, "value-with-different-size")
    second = build_dataset_manifest(tmp_path, chain="ethereum", hash_index_path=index)
    assert second.file_hashes["ethereum/transactions.csv"] != first_digest
    assert second.file_hashes["ethereum/transactions.csv"] == hashlib.sha256(source.read_bytes()).hexdigest()
    cache = json.loads(index.read_text(encoding="utf-8"))
    assert str(source.resolve()) in cache
    assert {"path", "size", "mtime_ns", "inode", "sha256"} <= set(cache[str(source.resolve())])


def test_partial_manifest_is_explicitly_truncated(tmp_path: Path) -> None:
    _write_transactions(tmp_path / "bsc" / "a.csv")
    _write_transactions(tmp_path / "bsc" / "b.csv")
    manifest = build_dataset_manifest(tmp_path, chain="bsc", max_files=1)
    assert manifest.manifest_complete is False
    assert manifest.truncated is True
    assert manifest.max_files_requested == 1
    assert manifest.files_discovered == 2
    assert manifest.files_processed == 1
    assert manifest.files_failed == 0


def test_full_manifest_records_label_semantics(tmp_path: Path) -> None:
    _write_transactions(tmp_path / "polygon" / "a.csv")
    labels = tmp_path / "labels.csv"
    labels.write_text("Chain,Contract,Category\npolygon,x,0\npolygon,y,6\n", encoding="utf-8")
    manifest = build_dataset_manifest(tmp_path, chain="polygon", labels_path=labels)
    assert manifest.manifest_complete is True
    assert manifest.label_mapping == {"0": "benign", "nonzero": "fraud", "missing": "unlabeled"}
    assert manifest.label_categories == {"0": 1, "6": 1}


def test_full_manifest_hashes_auxiliary_source_files(tmp_path: Path) -> None:
    _write_transactions(tmp_path / "bsc" / "a.csv")
    auxiliary = tmp_path / "bsc" / "list.txt"
    auxiliary.write_text("a.csv\n", encoding="utf-8")

    manifest = build_dataset_manifest(tmp_path, chain="bsc")

    assert manifest.files_discovered == 2
    assert len(manifest.file_hashes) == 2
    assert manifest.file_hashes["bsc/list.txt"] == hashlib.sha256(auxiliary.read_bytes()).hexdigest()
    assert manifest.hash_verification == "full"
