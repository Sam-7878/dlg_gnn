"""Fail-closed Round 4 SCI main-track paper-readiness gate."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.round4.artifact_paths import (
    CHECKPOINT_MANIFEST_DIR, DATASET_DIR, RAW_PREDICTION_DIR, RESULTS_DIR,
    RISK_CHECKPOINT_DIR, ensure_dirs,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate() -> dict:
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    manifest_path = DATASET_DIR / "real_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    checks["dataset_paper_eligible"] = manifest.get("paper_eligible") is True
    checks["chronological_real_split"] = manifest.get("split_type") == "chronological_real"
    checks["recorded_timestamp"] = manifest.get("timestamp_source") == "recorded_transaction_timestamp"
    checks["auditable_graph_source"] = bool(manifest.get("graph_source") and manifest.get("upstream_manifest_sha256"))
    checks["context_safe_for_main"] = manifest.get("context_policy") == "excluded_from_main_detector_metrics"
    checks["future_edges_zero"] = manifest.get("future_edge_count") == 0
    checks["class_support_all_splits"] = all(
        manifest.get("split", {}).get(name, {}).get("n_positive", 0) > 0
        and manifest.get("split", {}).get(name, {}).get("n_negative", 0) > 0
        for name in ("train", "validation", "test")
    )
    training_path = CHECKPOINT_MANIFEST_DIR / "training_summary.json"
    training = json.loads(training_path.read_text()) if training_path.is_file() else {}
    checks["real_gnn_five_seeds"] = (
        training.get("gnn_source") == "real_checkpoint" and training.get("seed_count", 0) >= 5
        and set(training.get("seeds", [])) >= {7, 17, 27, 37, 47}
    )
    risk_path = RISK_CHECKPOINT_DIR / "training_summary.json"
    risk = json.loads(risk_path.read_text()) if risk_path.is_file() else {}
    checks["trained_risk_encoder_provenance"] = (
        risk.get("seed_count", 0) >= 5 and all(not run.get("test_accessed", True) for run in risk.get("runs", []))
        and all(bool(run.get("checkpoint_sha256")) for run in risk.get("runs", []))
    )
    prediction_manifest_path = RESULTS_DIR / "raw_prediction_manifest.json"
    prediction_manifest = json.loads(prediction_manifest_path.read_text()) if prediction_manifest_path.is_file() else {}
    entries = prediction_manifest.get("entries", [])
    checks["raw_predictions_complete"] = (
        prediction_manifest.get("seed_count", 0) >= 5
        and set(prediction_manifest.get("passes", [])) >= {1, 5, 10, 20, 30}
        and len(entries) >= 25
    )
    checks["raw_prediction_hashes_valid"] = bool(entries) and all(
        (ROOT / entry["raw_predictions"]).is_file()
        and sha256(ROOT / entry["raw_predictions"]) == entry.get("raw_predictions_sha256")
        and bool(entry.get("checkpoint_sha256")) for entry in entries
    )
    main_path = RESULTS_DIR / "main_results.csv"
    controlled_path = RESULTS_DIR / "controlled_context_results.csv"
    if main_path.is_file() and controlled_path.is_file():
        main = pd.read_csv(main_path); controlled = pd.read_csv(controlled_path)
        checks["main_controlled_separated"] = (
            set(main["track"]) == {"SCI Main Track"} and not main["context_used"].astype(bool).any()
            and set(controlled["track"]) == {"Controlled Context-Augmentation Study"}
            and not controlled["paper_eligible"].astype(bool).any()
        )
    else:
        checks["main_controlled_separated"] = False
    for name, passed in checks.items():
        if not passed:
            reasons.append(name)
    return {
        "gate_version": "round4-paper-ready-v2.0", "paper_ready": all(checks.values()),
        "track": "SCI Main Track", "checks": checks, "failed_checks": reasons,
        "controlled_track_promoted": False,
        "source_availability_disclosure": manifest.get("source_availability"),
        "license_disclosure": manifest.get("license_availability"),
    }


def main() -> int:
    ensure_dirs(); result = evaluate()
    (RESULTS_DIR / "paper_ready_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["paper_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
