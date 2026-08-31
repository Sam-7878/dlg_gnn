#!/usr/bin/env python3
"""Run source-only transfer on each target chain's frozen temporal test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from gog_fraud.pipelines.run_round4_experiments import SciV2Records, _dlg_scores, _fit_dlg
from validation.sci_v3_final_common import (
    atomic_csv,
    atomic_json,
    binary_metrics,
    select_f1_threshold,
    sha256_file,
)


def normalize_source(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1
    return tuple(((item - mean) / scale).astype(np.float32) for item in (train, *others))


def run(config_path: Path, output_dir: Path) -> pd.DataFrame:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset_root"])
    dataset = SciV2Records(dataset_root, chain_feature=bool(config["chain_id_feature"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for transfer in config["transfers"]:
        sources = tuple(transfer["sources"])
        target = str(transfer["target"])
        train_ids = [sample for chain in sources for sample in dataset.ids(chain, "train")]
        valid_ids = [sample for chain in sources for sample in dataset.ids(chain, "validation")]
        test_ids = dataset.ids(target, "test")
        train_x, train_y = dataset.arrays(train_ids)
        valid_x, valid_y = dataset.arrays(valid_ids)
        test_x, test_y = dataset.arrays(test_ids)
        train_x, valid_x, test_x = normalize_source(train_x, valid_x, test_x)
        times = [int(dataset.records[sample]["event_end"]) for sample in test_ids]

        for seed in config["seeds"]:
            model, fit_meta = _fit_dlg(
                train_x,
                train_y,
                variant=str(config["model"]),
                seed=int(seed),
                epochs=int(config["epochs"]),
                device=device,
            )
            valid_score, _, _ = _dlg_scores(model, train_x, valid_x, device)
            threshold = select_f1_threshold(valid_y, valid_score)
            test_score, _, _ = _dlg_scores(model, train_x, test_x, device)
            name = f"{'+'.join(sources)}__to__{target}__seed{seed}.csv"
            prediction_path = prediction_dir / name
            atomic_csv(
                prediction_path,
                pd.DataFrame(
                    {
                        "sample_id": test_ids,
                        "label": test_y,
                        "score": test_score,
                        "event_end": times,
                    }
                ),
            )
            rows.append(
                {
                    "protocol": "strict_target_temporal_holdout",
                    "train_chains": "+".join(sources),
                    "target_chain": target,
                    "seed": int(seed),
                    "target_excluded_from_fit": True,
                    "preprocessing_fit_scope": "source_train_only",
                    "threshold_selection_scope": "source_validation_only",
                    "target_test_start": min(times),
                    "target_test_end": max(times),
                    "threshold": threshold,
                    "prediction_artifact": str(prediction_path),
                    "prediction_sha256": sha256_file(prediction_path),
                    "fitted_state_hash": fit_meta["fitted_state_hash"],
                    "metric_undefined_reason": (
                        "target temporal holdout contains no fraud-positive samples; ROC-AUC, PR-AUC, MCC, and fraud recall are mathematically undefined"
                        if int(test_y.sum()) == 0
                        else ""
                    ),
                    **binary_metrics(test_y, test_score, threshold),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    frame = pd.DataFrame(rows).sort_values(["target_chain", "seed"])
    atomic_csv(output_dir / "cross_chain_temporal_strict_metrics.csv", frame)
    atomic_json(
        output_dir / "protocol_manifest.json",
        {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "dataset_split_hashes": dataset.split_hashes,
            "device": str(device),
            "records": len(frame),
        },
    )
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sci_v3_final/cross_chain_temporal_strict.yaml")
    parser.add_argument("--output-dir", default="results/sci_v3_final/cross_chain")
    args = parser.parse_args()
    frame = run(Path(args.config).resolve(), Path(args.output_dir).resolve())
    print(json.dumps({"records": len(frame), "targets": sorted(frame.target_chain.unique())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
