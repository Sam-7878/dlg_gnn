"""Fail-closed SCI paper-readiness validation for GraphRAG Round 3."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

LOCAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOCAL_ROOT))

from experiments.round3.artifact_paths import (
    CHECKPOINT_DIR,
    CHECKPOINT_MANIFEST_DIR,
    RAW_PREDICTION_DIR,
    ROOT,
    ROUND3_RESULTS,
    ensure_round3_artifact_dirs,
)


REAL_TIMESTAMP_SOURCES = {"raw_transaction_timestamp", "on_chain_transaction_timestamp"}


def _load_json(path: Path, failures: list[str]) -> dict:
    if not path.is_file():
        failures.append(f"missing artifact: {path.relative_to(ROOT)}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_paper_ready(expected_seeds: Iterable[int] = (7, 17, 27, 37, 47)) -> dict:
    seeds = tuple(int(seed) for seed in expected_seeds)
    failures: list[str] = []
    dataset = _load_json(ROUND3_RESULTS / "real_dataset_manifest.json", failures)
    summary_path = CHECKPOINT_MANIFEST_DIR / "training_summary_v3.json"
    summary = _load_json(summary_path, failures)

    if dataset:
        if dataset.get("split_type") != "chronological_real":
            failures.append("dataset split_type must be chronological_real")
        if dataset.get("timestamp_source") not in REAL_TIMESTAMP_SOURCES:
            failures.append("split must originate from recorded transaction timestamps")
        if dataset.get("graph_source") not in {"real_gog", "gog_sci_v2"}:
            failures.append("graph_source must identify an auditable real GoG derivative")
        if dataset.get("paper_eligible") is not True:
            failures.append("dataset manifest is not marked paper_eligible=true")
        if int(dataset.get("test_fraud", 0)) <= 0:
            failures.append("chronological test split has no positive support")
        if int(dataset.get("test_size", 0)) - int(dataset.get("test_fraud", 0)) <= 0:
            failures.append("chronological test split has no negative support")
        if dataset.get("context_source") == "label_conditioned_synthetic":
            failures.append("label-conditioned context cannot enter paper-ready detector metrics")

    if summary:
        if summary.get("gnn_source") != "real_checkpoint":
            failures.append("training summary gnn_source must be real_checkpoint")
        if summary.get("split_type") != "chronological_real":
            failures.append("training summary split_type must be chronological_real")
        completed = {int(seed) for seed in summary.get("seeds", [])}
        missing = sorted(set(seeds) - completed)
        if missing:
            failures.append(f"missing trained seeds: {missing}")
        if int(summary.get("seed_count", 0)) < len(seeds):
            failures.append(f"seed_count must be at least {len(seeds)}")
        if dataset and summary.get("dataset_sha256") != dataset.get("graph_sha256"):
            failures.append("training and dataset hashes do not match")

    for seed in seeds:
        manifest_path = CHECKPOINT_MANIFEST_DIR / f"l1v3_seed{seed}.json"
        checkpoint_manifest = _load_json(manifest_path, failures)
        if checkpoint_manifest:
            checkpoint_path = ROOT / checkpoint_manifest.get("checkpoint_path", "")
            if not checkpoint_path.is_file():
                checkpoint_path = CHECKPOINT_DIR / Path(
                    checkpoint_manifest.get("checkpoint_path", "")
                ).name
            expected_hash = checkpoint_manifest.get("checkpoint_sha256", "")
            if not checkpoint_path.is_file():
                failures.append(f"missing checkpoint for seed {seed}")
            elif len(expected_hash) != 64 or _sha256(checkpoint_path) != expected_hash:
                failures.append(f"checkpoint hash mismatch for seed {seed}")
        prediction = RAW_PREDICTION_DIR / f"seed{seed}_T10_preds.csv"
        if not prediction.is_file() or prediction.stat().st_size == 0:
            failures.append(f"missing raw T=10 predictions for seed {seed}")

    latency = ROUND3_RESULTS / "real_e2e_latency.csv"
    if not latency.is_file() or latency.stat().st_size == 0:
        failures.append("missing real_e2e_latency.csv")

    return {
        "paper_ready": not failures,
        "expected_seeds": list(seeds),
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = evaluate_paper_ready(int(value) for value in args.seeds.split(","))
    if args.write_report:
        ensure_round3_artifact_dirs()
        (ROUND3_RESULTS / "paper_ready_gate.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))
    return 0 if report["paper_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
