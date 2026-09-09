import hashlib
import json

from experiments.round6.recovery import audit_recovery


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recovery_requires_all_exact_hashes(tmp_path):
    preserved = tmp_path / "preserved.json"
    dataset = tmp_path / "dataset"
    upstream = tmp_path / "upstream"
    dataset.mkdir()
    (dataset / "graph.pt").write_bytes(b"graph")
    (dataset / "transactions.parquet").write_bytes(b"transactions")
    (dataset / "split_manifest.json").write_bytes(b"split")
    (dataset / "future_edge_audit.csv").write_bytes(b"audit")
    manifest = {
        "dataset_name": "fixture", "dataset_version": "v1",
        "graph_sha256": _digest(dataset / "graph.pt"),
        "transactions_sha256": _digest(dataset / "transactions.parquet"),
        "split_manifest_sha256": _digest(dataset / "split_manifest.json"),
        "future_edge_audit_sha256": _digest(dataset / "future_edge_audit.csv"),
        "upstream_manifest_sha256": {}, "future_edge_count": 0, "graph_hash_failures": 0,
    }
    preserved.write_text(json.dumps(manifest))
    (dataset / "real_dataset_manifest.json").write_bytes(preserved.read_bytes())
    result = audit_recovery(
        dataset, upstream, preserved, tmp_path / "result.json", tmp_path / "report.md"
    )
    assert result["dataset_exact_recovery"] is True
    assert result["future_edge_audit_verified"] is True

    (dataset / "future_edge_audit.csv").write_bytes(b"changed")
    result = audit_recovery(
        dataset, upstream, preserved, tmp_path / "result2.json", tmp_path / "report2.md"
    )
    assert result["dataset_exact_recovery"] is False
    assert result["future_edge_audit_verified"] is False

