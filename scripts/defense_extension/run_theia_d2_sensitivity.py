"""Artifact-level THEIA no-total-events sensitivity for Defense Round D2.

This does not replace or mutate any D1 result. The underlying D1 artifact is
synthetic, so these runs are diagnostic evidence about that artifact only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from gog_fraud.extensions.defense.defense_registry import load_defense_dataset
from run_defense_multiseed import instantiate_detector, stratified_split_indices

MODELS = ("DOMINANT", "DLG-Base", "DLG-Aug")
SEEDS = (42, 43, 44, 45, 46)
REMOVED_FEATURE_INDEX = 13
REMOVED_FEATURE_NAME = "total_events_log1p"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark/sci_defense_extension.yaml")
    parser.add_argument("--output-dir", default="outputs/sci_defense_extension/d2/sensitivity")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path("outputs/sci_defense_extension/processed/darpa_theia/darpa_tc_theia_e3.pt")
    original = load_defense_dataset("DARPA-TC-THEIA")
    keep = [i for i in range(original.x.size(1)) if i != REMOVED_FEATURE_INDEX]
    variant = original.clone()
    variant.x = original.x[:, keep].contiguous()

    records: list[dict] = []
    for model in MODELS:
        for seed in SEEDS:
            result_path = raw_dir / f"THEIA-no-total-events__{model}__seed{seed}.json"
            if result_path.exists():
                records.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
            data = variant.clone()
            y = data.y.cpu().numpy().astype(np.int64)
            _, val_idx, test_idx = stratified_split_indices(y, seed, 0.2, 0.2)
            detector = instantiate_detector(model, config, gpu=0 if torch.cuda.is_available() else -1)
            started = time.perf_counter()
            detector.fit(data)
            elapsed = time.perf_counter() - started
            scores = detector.decision_score_
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            scores = np.asarray(scores).reshape(-1)
            thresholds = np.unique(np.quantile(scores[val_idx], np.linspace(0, 1, min(201, len(val_idx)))))
            threshold = float(thresholds[int(np.argmax([
                f1_score(y[val_idx], scores[val_idx] >= value, zero_division=0)
                for value in thresholds
            ]))])
            record = {
                "dataset": "THEIA-no-total-events",
                "evidence_scope": "d1_synthetic_artifact_diagnostic_only",
                "model": model,
                "seed": seed,
                "status": "success",
                "removed_feature": REMOVED_FEATURE_NAME,
                "configured_epochs": int(config["training"]["epochs"]),
                "roc_auc": float(roc_auc_score(y[test_idx], scores[test_idx])),
                "pr_auc": float(average_precision_score(y[test_idx], scores[test_idx])),
                "validation_f1": float(f1_score(y[test_idx], scores[test_idx] >= threshold, zero_division=0)),
                "validation_threshold": threshold,
                "fit_seconds": elapsed,
            }
            result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            records.append(record)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    frame = pd.DataFrame(records).sort_values(["model", "seed"])
    raw_csv = output / "theia_no_total_events_raw.csv"
    frame.to_csv(raw_csv, index=False)
    summary = frame.groupby("model", as_index=False)[["roc_auc", "pr_auc", "validation_f1"]].agg(["mean", "std"])
    summary.columns = ["_".join(filter(None, col)) for col in summary.columns]
    summary.to_csv(output / "theia_no_total_events_summary.csv", index=False)
    manifest = {
        "status": "complete",
        "runs": len(frame),
        "expected_runs": len(MODELS) * len(SEEDS),
        "source_artifact_sha256": sha256_file(source_path),
        "variant_feature_sha256": tensor_hash(variant.x),
        "graph_unchanged": torch.equal(original.edge_index, variant.edge_index),
        "labels_unchanged": torch.equal(original.y, variant.y),
        "removed_feature_index": REMOVED_FEATURE_INDEX,
        "removed_feature_name": REMOVED_FEATURE_NAME,
        "interpretation_limit": "The D1 input is synthetic; results are not official THEIA evidence.",
        "raw_csv_sha256": sha256_file(raw_csv),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
