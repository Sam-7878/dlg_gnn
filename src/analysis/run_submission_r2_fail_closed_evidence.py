"""Fail-closed SCI-v3 R2 evidence runner for an unavailable frozen dataset root.

The frozen graph cache is sufficient to re-evaluate the production GIN/GATv2
path.  Tabular validation features are not recoverable from that cache, so this
runner records FAIL-C and never uses test labels to tune their cascades.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch_geometric.loader import DataLoader

from gog_fraud.production.calibrated_cascade import (
    PlattLogOddsCalibrator,
    apply_cascade,
    select_cascade_on_validation,
    select_f1_threshold,
)
from gog_fraud.production.closure import infer_level1, infer_level2, load_seed_bundle, relation_data
from validation.sci_v3_final_common import atomic_csv, atomic_json, binary_metrics, sha256_file


def _support(labels: np.ndarray) -> dict[str, Any]:
    positive = int(labels.sum())
    return {"N": len(labels), "N_positive": positive, "N_negative": len(labels) - positive,
            "metric_defined": 0 < positive < len(labels), "undefined_reason": ""}


def _ece(labels: np.ndarray, scores: np.ndarray) -> float:
    value = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (scores >= lower) & (scores < upper if upper < 1 else scores <= upper)
        if mask.any():
            value += mask.mean() * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return float(value)


def _calibration_rows(model: str, seed: int, split: str, labels: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for method, score in (("raw", raw), ("platt_log_odds", calibrated)):
        rows.append({"model": model, "seed": seed, "split": split, "calibration": method,
                     "brier": brier_score_loss(labels, score),
                     "log_loss": log_loss(labels, np.c_[1 - score, score], labels=[0, 1]),
                     "ece_10": _ece(labels, score), "roc_auc": roc_auc_score(labels, score), **_support(labels)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sci_v3_submission_r2/closure.yaml"))
    parser.add_argument("--cache", type=Path, default=Path("results/sci_v3_submission/cache/bounded_graphs.pt"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(Path(cfg["base_config"]).read_text(encoding="utf-8"))
    source, output = Path(cfg["source_results"]), Path(cfg["output_root"])
    prediction_root = output / "cascade/predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(base_cfg["dataset_root"])
    if dataset_root.exists():
        raise RuntimeError("dataset root is available; run run_submission_r2_calibrated_cascades.py instead")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(base_cfg["level1"]["batch_size"])
    epsilon = float(cfg["calibration"]["epsilon"])
    c_value = float(cfg["calibration"]["logistic_c"])
    metrics, calibration, traces, distributions, acceptance = [], [], [], [], []
    test_ids = [row["sample_id"] for row in cache["metadata"]["test"]]

    for seed in map(int, cfg["seeds"]):
        level1, level2, _, _ = load_seed_bundle(source / f"checkpoints/seed{seed}", device)
        inferred = {split: infer_level1(level1, DataLoader(cache["graphs"][split], batch_size=batch_size, shuffle=False), device)
                    for split in ("train", "validation", "test")}
        train_score, train_y, train_embedding = inferred["train"]
        valid_score, valid_y, valid_embedding = inferred["validation"]
        test_score, test_y, test_embedding = inferred["test"]
        valid_relation = relation_data(train_embedding, train_score, train_y, valid_embedding, valid_score, valid_y,
                                       int(base_cfg["level2"]["knn_k"]))
        test_relation = relation_data(train_embedding, train_score, train_y, test_embedding, test_score, test_y,
                                      int(base_cfg["level2"]["knn_k"]))
        raw_valid_deep = infer_level2(level2, valid_relation, len(train_y), device)
        raw_test_deep = infer_level2(level2, test_relation, len(train_y), device)
        fast_calibrator = PlattLogOddsCalibrator.fit(valid_score, valid_y, logistic_c=c_value, epsilon=epsilon, random_state=seed)
        deep_calibrator = PlattLogOddsCalibrator.fit(raw_valid_deep, valid_y, logistic_c=c_value, epsilon=epsilon, random_state=seed)
        valid_fast, test_fast = fast_calibrator.transform(valid_score), fast_calibrator.transform(test_score)
        valid_deep, test_deep = deep_calibrator.transform(raw_valid_deep), deep_calibrator.transform(raw_test_deep)
        selection = select_cascade_on_validation(valid_y, valid_fast, valid_deep,
            fast_weight_grid=cfg["calibration"]["fast_weight_grid"],
            deep_budget_grid=cfg["calibration"]["deep_budget_grid"], epsilon=epsilon)
        final, routed, fused = apply_cascade(test_fast, test_deep, selection, epsilon=epsilon)
        fast_threshold = select_f1_threshold(valid_y, valid_fast)
        fast_metrics = binary_metrics(test_y, test_fast, fast_threshold)
        cascade_metrics = binary_metrics(test_y, final, selection.final_threshold)
        common = {"seed": seed, "measurement_type": "measured_prediction", **_support(test_y)}
        metrics.extend([
            {"model": "ProductionLevel1GIN", "routing_policy": "only", "threshold": fast_threshold,
             "deep_route_rate": 0.0, "direct_exit_rate": 1.0, **common, **fast_metrics},
            {"model": "ProductionLevel1GIN->ProductionLevel2GATv2", "routing_policy": "validation_calibrated",
             "threshold": selection.final_threshold, "deep_route_rate": routed.mean(), "direct_exit_rate": 1-routed.mean(),
             **asdict(selection), **common, **cascade_metrics},
        ])
        calibration.extend(_calibration_rows("ProductionLevel1GIN", seed, "validation", valid_y, valid_score, valid_fast))
        calibration.extend(_calibration_rows("ProductionLevel1GIN", seed, "test", test_y, test_score, test_fast))
        calibration.extend(_calibration_rows("ProductionLevel2GATv2", seed, "validation", valid_y, raw_valid_deep, valid_deep))
        calibration.extend(_calibration_rows("ProductionLevel2GATv2", seed, "test", test_y, raw_test_deep, test_deep))
        traces.append({"seed": seed, "fast_model": "ProductionLevel1GIN", "interface_case": "A",
                       "fast_output_semantics": "probability", "deep_output_semantics": "probability",
                       "calibration_input_space": "log_odds", "fusion_space": "calibrated_log_odds",
                       "fast_calibrator": json.dumps(fast_calibrator.to_dict(), sort_keys=True),
                       "deep_calibrator": json.dumps(deep_calibrator.to_dict(), sort_keys=True), **asdict(selection)})
        for split, labels, values in (("validation", valid_y, {"fast_raw": valid_score, "fast_calibrated": valid_fast,
                                      "deep_raw": raw_valid_deep, "deep_calibrated": valid_deep}),
                                      ("test", test_y, {"fast_raw": test_score, "fast_calibrated": test_fast,
                                      "deep_raw": raw_test_deep, "deep_calibrated": test_deep})):
            for stage, score in values.items():
                for label in (0, 1):
                    selected = score[labels == label]
                    distributions.append({"seed": seed, "model": "ProductionLevel1GIN", "split": split,
                        "stage": stage, "label": label, "N": len(selected), "mean": np.mean(selected),
                        "std": np.std(selected), "q01": np.quantile(selected,.01), "q10": np.quantile(selected,.1),
                        "q50": np.quantile(selected,.5), "q90": np.quantile(selected,.9), "q99": np.quantile(selected,.99)})
        atomic_csv(prediction_root / f"ProductionLevel1GIN__seed{seed}.csv", pd.DataFrame({
            "sample_id": test_ids, "label": test_y, "raw_fast_score": test_score,
            "calibrated_fast_score": test_fast, "raw_deep_score": raw_test_deep,
            "calibrated_deep_score": test_deep, "fusion_score": fused, "final_score": final,
            "deep_executed": routed.astype(int), "fast_prediction": (test_fast >= fast_threshold).astype(int),
            "final_prediction": (final >= selection.final_threshold).astype(int)}))
        acceptance.append({"seed": seed, "model": "ProductionLevel1GIN", "fast_f1": fast_metrics["f1"],
                           "cascade_f1": cascade_metrics["f1"], "delta_vs_fast": cascade_metrics["f1"]-fast_metrics["f1"],
                           "deep_route_rate": routed.mean(), "evidence_status": "MEASURED"})
        for model in ("XGBoostFastTriage", "LightGBMFastTriage"):
            acceptance.append({"seed": seed, "model": model, "fast_f1": np.nan, "cascade_f1": np.nan,
                               "delta_vs_fast": np.nan, "deep_route_rate": np.nan,
                               "evidence_status": "UNAVAILABLE_MISSING_FROZEN_VALIDATION_FEATURES"})
        print(f"calibrated seed={seed}", flush=True)

    atomic_csv(output / "cascade/calibrated_cascade_metrics.csv", pd.DataFrame(metrics))
    atomic_csv(output / "cascade/calibration_metrics.csv", pd.DataFrame(calibration))
    atomic_csv(output / "cascade/interface_trace.csv", pd.DataFrame(traces))
    atomic_csv(output / "cascade/score_distributions.csv", pd.DataFrame(distributions))
    atomic_csv(output / "cascade/acceptance_gate_by_seed.csv", pd.DataFrame(acceptance))
    atomic_json(output / "cascade/acceptance_gate.json", {"gate": "FAIL-C",
        "interpretation": "tabular cascades removed: frozen validation features unavailable and test tuning forbidden",
        "production_gin_path_status": "MEASURED", "interface_case": "A", "missing_input": str(dataset_root)})
    atomic_json(output / "cascade/leakage_audit.json", {"status": "PASS_WITH_FAIL_C_TABULAR_REMOVAL",
        "selection_inputs": ["validation labels", "validation GIN score", "validation GATv2 score"],
        "test_labels_used_for_selection": False, "fallback_action": "remove tabular cascade claims",
        "config_sha256": sha256_file(args.config), "cache_sha256": sha256_file(args.cache)})
    print(json.dumps({"gate": "FAIL-C", "production_rows": len(metrics), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
