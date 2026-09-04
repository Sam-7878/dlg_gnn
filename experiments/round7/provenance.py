"""Fail-closed provenance helpers for the final dataset reacquisition round."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


SEEDS = (7, 17, 27, 37, 47)
EXPECTED_PACKED_HASHES = {
    "graph.pt": "067cbdd7d7c055da91dbed9c492ad5a099c35e178f718e53f9dfdabab908b1cd",
    "transactions.parquet": "4d240fe8d5488f6f27fd1d475d039abfc96aa40aa6dc34d34a75dfa92be3df3d",
    "split_manifest.json": "7f388c5163293f5706cb07427747b1fc6988ae749c8d1a0e104e79b6d83accfa",
    "future_edge_audit.csv": "395cc4fe3c0c2198fbb25368f9cf843bd9de4352efcbcba8a7d8e77fd5e43f7f",
}
EXPECTED_UPSTREAM_MANIFEST_HASHES = {
    "ethereum": "1efed3a8977f56cc30bd79c97f95eee57b825fe25971e7f72b2e0a93402ae2be",
    "bsc": "edcf3890377a985b8c03da68fda5237590e878672c2d6ac31b784c1ac1ef60b7",
    "polygon": "d8948abd672a1abe68ae11229b4d482c8d8415ae3cfddff76d48cb776d63ce0c",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_once(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = sha256_file(source)
    if destination.exists():
        archived_hash = sha256_file(destination)
        if archived_hash != source_hash:
            raise RuntimeError(f"immutable archive collision: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        archived_hash = sha256_file(destination)
    if archived_hash != source_hash:
        raise RuntimeError(f"archive hash mismatch: {destination}")
    return {
        "source": str(source),
        "archived": str(destination),
        "sha256": archived_hash,
        "bytes": destination.stat().st_size,
    }


def _archive_plan(root: Path) -> Iterable[tuple[Path, Path]]:
    round4 = root / "results/graphrag/round_4"
    final = root / "results/main_final"
    for seed in SEEDS:
        yield round4 / f"real_checkpoints/seed{seed}.pt", Path(f"checkpoints/seed{seed}.pt")
        yield round4 / f"checkpoint_manifests/seed{seed}.json", Path(f"checkpoint_manifests/seed{seed}.json")
        for passes in (1, 10):
            name = f"seed{seed}_T{passes}.csv"
            yield final / "raw_predictions" / name, Path("raw_predictions") / name
    for name in (
        "checkpoints_manifest.json",
        "ensemble_predictions.parquet",
        "model_metrics.csv",
        "statistical_comparisons.csv",
        "temporal_slice_metrics.csv",
        "latency_definition.json",
        "dataset_recovery_manifest.json",
    ):
        yield final / name, Path(name)
    yield root / "results/paper_ready_gate_v7.json", Path("paper_ready_gate_v7.json")
    yield final / "raw_predictions/manifest.json", Path("raw_predictions/manifest.json")


def create_preserved_archive(root: Path, archive: Path) -> dict[str, Any]:
    """Create or verify the immutable v1 evidence archive.

    Existing bytes are never replaced. Re-running is verification-only for every
    file already present and fails if any source/archive digest differs.
    """
    manifest_path = archive / "archive_manifest.json"
    entries = [_copy_once(source, archive / relative) for source, relative in _archive_plan(root)]
    payload = {
        "archive_name": "gog_scimain_v1_preserved_panel",
        "policy": "copy-once; refuse overwrite on byte mismatch",
        "historical_only": True,
        "seed_count": len(SEEDS),
        "prediction_panel_count": len(SEEDS) * 2,
        "entries": entries,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"immutable archive manifest differs: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(encoded, encoding="utf-8")
    return payload


def verify_hash_contract(root: Path, expected: dict[str, str]) -> dict[str, Any]:
    entries = []
    for name, expected_hash in expected.items():
        path = root / name
        actual = sha256_file(path) if path.is_file() else None
        entries.append({
            "artifact": name,
            "path": str(path),
            "expected_sha256": expected_hash,
            "actual_sha256": actual,
            "match": actual == expected_hash,
        })
    return {"all_match": all(row["match"] for row in entries), "entries": entries}

