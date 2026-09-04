import json
from pathlib import Path

import pytest

from experiments.round7.provenance import create_preserved_archive, sha256_file, verify_hash_contract


def test_hash_contract_is_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "present.bin").write_bytes(b"known")
    expected = {"present.bin": sha256_file(tmp_path / "present.bin"), "missing.bin": "0" * 64}
    audit = verify_hash_contract(tmp_path, expected)
    assert audit["all_match"] is False
    assert [entry["match"] for entry in audit["entries"]] == [True, False]


def test_archive_is_copy_once_and_detects_collision(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    final = root / "results/main_final"
    round4 = root / "results/graphrag/round_4"
    for seed in (7, 17, 27, 37, 47):
        (round4 / "real_checkpoints").mkdir(parents=True, exist_ok=True)
        (round4 / "checkpoint_manifests").mkdir(parents=True, exist_ok=True)
        (round4 / f"real_checkpoints/seed{seed}.pt").write_bytes(f"ckpt-{seed}".encode())
        (round4 / f"checkpoint_manifests/seed{seed}.json").write_text("{}\n")
        (final / "raw_predictions").mkdir(parents=True, exist_ok=True)
        for passes in (1, 10):
            (final / f"raw_predictions/seed{seed}_T{passes}.csv").write_text("y,p\n0,0.1\n")
    for name in (
        "checkpoints_manifest.json", "ensemble_predictions.parquet", "model_metrics.csv",
        "statistical_comparisons.csv", "temporal_slice_metrics.csv", "latency_definition.json",
        "dataset_recovery_manifest.json",
    ):
        (final / name).write_text("{}\n")
    (final / "raw_predictions/manifest.json").write_text("{}\n")
    (root / "results/paper_ready_gate_v7.json").write_text("{}\n")

    archive = tmp_path / "archive"
    first = create_preserved_archive(root, archive)
    assert first["seed_count"] == 5
    assert len(first["entries"]) == 29
    assert json.loads((archive / "archive_manifest.json").read_text())["historical_only"] is True
    assert create_preserved_archive(root, archive) == first

    (archive / "checkpoints/seed7.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="immutable archive collision"):
        create_preserved_archive(root, archive)

