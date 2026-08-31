#!/usr/bin/env python3
"""Compare tabular and DLG fast paths under one validation-selected routing budget."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from gog_fraud.models.baselines.cascade import apply_budgeted_cascade, ambiguity_cutoff
from gog_fraud.pipelines.run_round4_experiments import SciV2Records, _dlg_scores, _fit_dlg, _normalize
from validation.sci_v3_final_common import SEEDS, atomic_csv, binary_metrics, select_f1_threshold, sha256_json


def run(dataset_root: Path, output_dir: Path, epochs: int, deep_budget: float) -> pd.DataFrame:
    dataset = SciV2Records(dataset_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raw_dir = output_dir / "predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for chain in ("ethereum", "bsc", "polygon", "pooled"):
        train_ids, valid_ids, test_ids = (dataset.ids(chain, group) for group in ("train", "validation", "test"))
        train_x, train_y = dataset.arrays(train_ids)
        valid_x, valid_y = dataset.arrays(valid_ids)
        test_x, test_y = dataset.arrays(test_ids)
        train_x, valid_x, test_x = _normalize(train_x, valid_x, test_x)
        positive_weight = float((len(train_y) - train_y.sum()) / max(1, train_y.sum()))

        for seed in SEEDS:
            full_model, _ = _fit_dlg(train_x, train_y, variant="DLG-Full-Fusion", seed=seed, epochs=epochs, device=device)
            full_val, _, _ = _dlg_scores(full_model, train_x, valid_x, device)
            full_test, _, full_latency = _dlg_scores(full_model, train_x, test_x, device)
            full_threshold = select_f1_threshold(valid_y, full_val)
            l1_model, _ = _fit_dlg(train_x, train_y, variant="DLG-L1", seed=seed, epochs=epochs, device=device)
            l1_val, _, _ = _dlg_scores(l1_model, train_x, valid_x, device)
            l1_test, _, l1_latency = _dlg_scores(l1_model, train_x, test_x, device)

            models = {
                "XGBoost": xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.05,
                    scale_pos_weight=positive_weight,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed,
                    tree_method="hist",
                    eval_metric="logloss",
                ),
                "LightGBM": lgb.LGBMClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.05,
                    scale_pos_weight=positive_weight,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed,
                    verbose=-1,
                ),
            }
            fast_outputs: dict[str, tuple[np.ndarray, np.ndarray, float]] = {"DLG-L1": (l1_val, l1_test, float(np.mean(l1_latency)))}
            for name, model in models.items():
                model.fit(train_x, train_y)
                validation_score = model.predict_proba(valid_x)[:, 1]
                started = time.perf_counter()
                test_score = model.predict_proba(test_x)[:, 1]
                latency_ms = (time.perf_counter() - started) * 1000.0
                fast_outputs[name] = (validation_score, test_score, latency_ms)

            for fast_name, (validation_score, test_score, fast_latency_ms) in fast_outputs.items():
                threshold = select_f1_threshold(valid_y, validation_score)
                cutoff = ambiguity_cutoff(validation_score, threshold, deep_budget)
                cascade_score, route_deep = apply_budgeted_cascade(test_score, full_test, threshold, cutoff)
                for mode, score, deep_mask in (
                    (f"{fast_name}-only", test_score, np.zeros(len(test_y), dtype=bool)),
                    (f"{fast_name}->Level2/Fusion", cascade_score, route_deep),
                ):
                    raw_path = raw_dir / f"{chain}__{mode.replace('/', '-') }__seed{seed}.csv"
                    atomic_csv(raw_path, pd.DataFrame({"sample_id": test_ids, "label": test_y, "score": score, "route_deep": deep_mask}))
                    deep_rate = float(deep_mask.mean())
                    modeled_cost = fast_latency_ms + deep_rate * float(np.mean(full_latency))
                    threshold_payload = {
                        "chain": chain,
                        "seed": seed,
                        "fast_path": fast_name,
                        "threshold": threshold,
                        "margin_cutoff": cutoff,
                        "deep_budget": deep_budget,
                        "selection_dataset": "validation",
                    }
                    rows.append(
                        {
                            "chain": chain,
                            "seed": seed,
                            "model": mode,
                            "fast_path": fast_name,
                            "deep_stage": "DLG-Full-Fusion",
                            "threshold": threshold,
                            "margin_cutoff": cutoff,
                            "routing_budget": deep_budget,
                            "deep_rate": deep_rate,
                            "selection_dataset": "validation",
                            "threshold_artifact_hash": sha256_json(threshold_payload),
                            "cost_accounting": "modeled_fast_plus_pdeep_times_full_batch_latency",
                            "modeled_e2e_cost_ms": modeled_cost,
                            "runtime_validated_selective_execution": False,
                            "prediction_artifact": str(raw_path),
                            **binary_metrics(test_y, score, threshold),
                        }
                    )
            del full_model, l1_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    atomic_csv(output_dir / "tabular_l2_cascade_metrics.csv", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/mnt/d/_Work/_data/GoG_sci_v2")
    parser.add_argument("--output-dir", default="results/sci_v3_final/baselines")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--deep-budget", type=float, default=0.42)
    args = parser.parse_args()
    frame = run(Path(args.dataset_root), Path(args.output_dir), args.epochs, args.deep_budget)
    print(json.dumps({"records": len(frame), "runtime_validated_selective_execution": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
