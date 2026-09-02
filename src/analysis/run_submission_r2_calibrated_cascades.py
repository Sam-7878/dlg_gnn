"""Rebuild the SCI-v3 submission cascades with validation-only calibration.

This script reuses the frozen round-109 checkpoints and dataset split manifests.
It writes only to ``results/sci_v3_submission_r2``.
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

from gog_fraud.pipelines.run_round4_experiments import SciV2Records
from gog_fraud.production.calibrated_cascade import (
    PlattLogOddsCalibrator,
    apply_cascade,
    select_cascade_on_validation,
    select_f1_threshold,
)
from gog_fraud.production.closure import build_graph_cache, infer_level1, infer_level2, load_seed_bundle, relation_data
from validation.sci_v3_final_common import atomic_csv, atomic_json, binary_metrics, sha256_file


def support(labels: np.ndarray) -> dict[str, Any]:
    positive = int(np.asarray(labels).sum())
    return {
        "N": int(len(labels)),
        "N_positive": positive,
        "N_negative": int(len(labels) - positive),
        "metric_defined": bool(0 < positive < len(labels)),
        "undefined_reason": "" if 0 < positive < len(labels) else "single_class_target",
    }


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (scores >= lower) & (scores < upper if upper < 1.0 else scores <= upper)
        if selected.any():
            value += selected.mean() * abs(float(labels[selected].mean()) - float(scores[selected].mean()))
    return float(value if total else np.nan)


def calibration_row(model: str, seed: int, split: str, labels: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for stage, scores in (("raw", raw), ("platt_log_odds", calibrated)):
        rows.append({
            "model": model,
            "seed": seed,
            "split": split,
            "calibration": stage,
            "brier": float(brier_score_loss(labels, scores)),
            "log_loss": float(log_loss(labels, np.c_[1.0 - scores, scores], labels=[0, 1])),
            "ece_10": expected_calibration_error(labels, scores),
            "roc_auc": float(roc_auc_score(labels, scores)),
            **support(labels),
        })
    return rows


def metric_row(
    model: str,
    seed: int,
    labels: np.ndarray,
    score: np.ndarray,
    threshold: float,
    routed: np.ndarray,
    selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "seed": seed,
        "threshold": float(threshold),
        "deep_route_rate": float(routed.mean()),
        "direct_exit_rate": float(1.0 - routed.mean()),
        "measurement_type": "measured_prediction",
        **support(labels),
        **binary_metrics(labels, score, threshold),
        **selection,
    }


def normalized_tabular_arrays(dataset: SciV2Records, normalization: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_ids, test_ids = (dataset.ids("pooled", split) for split in ("validation", "test"))
    valid_x, valid_y = dataset.arrays(valid_ids)
    test_x, test_y = dataset.arrays(test_ids)
    mean = np.asarray(normalization["mean"])
    scale = np.asarray(normalization["scale"])
    return (valid_x - mean) / scale, valid_y.astype(int), (test_x - mean) / scale, test_y.astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sci_v3_submission_r2/closure.yaml"))
    args = parser.parse_args()
    closure_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_path = Path(closure_cfg["base_config"])
    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    source = Path(closure_cfg["source_results"])
    output = Path(closure_cfg["output_root"])
    prediction_root = output / "cascade/predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = build_graph_cache(
        Path(base_cfg["dataset_root"]),
        source / "cache/bounded_graphs.pt",
        int(base_cfg["bounded_graph"]["max_edges"]),
        int(base_cfg["bounded_graph"]["max_nodes"]),
    )
    dataset = SciV2Records(Path(base_cfg["dataset_root"]))
    batch_size = int(base_cfg["level1"]["batch_size"])
    test_ids = [row["sample_id"] for row in cache["metadata"]["test"]]
    if list(dataset.ids("pooled", "test")) != test_ids:
        raise RuntimeError("tabular and graph test order differ; refusing an invalid paired comparison")

    metrics: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []
    acceptance: list[dict[str, Any]] = []
    old_metrics = pd.read_csv(source / "baselines/production_backbone_metrics.csv")
    epsilon = float(closure_cfg["calibration"]["epsilon"])
    logistic_c = float(closure_cfg["calibration"]["logistic_c"])

    for seed in map(int, closure_cfg["seeds"]):
        level1, level2, tabular, metadata = load_seed_bundle(source / f"checkpoints/seed{seed}", device)
        inferred = {}
        for split in ("train", "validation", "test"):
            inferred[split] = infer_level1(
                level1,
                DataLoader(cache["graphs"][split], batch_size=batch_size, shuffle=False),
                device,
            )
        train_score, train_y, train_embedding = inferred["train"]
        valid_score, valid_y, valid_embedding = inferred["validation"]
        test_score, test_y, test_embedding = inferred["test"]
        valid_relation = relation_data(
            train_embedding, train_score, train_y, valid_embedding, valid_score, valid_y, int(base_cfg["level2"]["knn_k"])
        )
        test_relation = relation_data(
            train_embedding, train_score, train_y, test_embedding, test_score, test_y, int(base_cfg["level2"]["knn_k"])
        )
        raw_valid_deep = infer_level2(level2, valid_relation, len(train_y), device)
        raw_test_deep = infer_level2(level2, test_relation, len(train_y), device)
        deep_calibrator = PlattLogOddsCalibrator.fit(
            raw_valid_deep, valid_y, logistic_c=logistic_c, epsilon=epsilon, random_state=seed
        )
        valid_deep = deep_calibrator.transform(raw_valid_deep)
        test_deep = deep_calibrator.transform(raw_test_deep)
        calibration.extend(calibration_row("ProductionLevel2GATv2", seed, "validation", valid_y, raw_valid_deep, valid_deep))
        calibration.extend(calibration_row("ProductionLevel2GATv2", seed, "test", test_y, raw_test_deep, test_deep))

        valid_x, tab_valid_y, test_x, tab_test_y = normalized_tabular_arrays(dataset, tabular["normalization"])
        if not np.array_equal(valid_y, tab_valid_y) or not np.array_equal(test_y, tab_test_y):
            raise RuntimeError("tabular and graph labels differ; paired cascade comparison is invalid")
        raw_fast = {
            "XGBoostFastTriage": (tabular["XGBoostFastTriage"].predict_proba(valid_x)[:, 1], tabular["XGBoostFastTriage"].predict_proba(test_x)[:, 1]),
            "LightGBMFastTriage": (tabular["LightGBMFastTriage"].predict_proba(valid_x)[:, 1], tabular["LightGBMFastTriage"].predict_proba(test_x)[:, 1]),
            "ProductionLevel1GIN": (valid_score, test_score),
        }
        for fast_name, (raw_valid_fast, raw_test_fast) in raw_fast.items():
            fast_calibrator = PlattLogOddsCalibrator.fit(
                raw_valid_fast, valid_y, logistic_c=logistic_c, epsilon=epsilon, random_state=seed
            )
            valid_fast = fast_calibrator.transform(raw_valid_fast)
            test_fast = fast_calibrator.transform(raw_test_fast)
            calibration.extend(calibration_row(fast_name, seed, "validation", valid_y, raw_valid_fast, valid_fast))
            calibration.extend(calibration_row(fast_name, seed, "test", test_y, raw_test_fast, test_fast))
            selection = select_cascade_on_validation(
                valid_y,
                valid_fast,
                valid_deep,
                fast_weight_grid=closure_cfg["calibration"]["fast_weight_grid"],
                deep_budget_grid=closure_cfg["calibration"]["deep_budget_grid"],
                epsilon=epsilon,
            )
            final, routed, fused = apply_cascade(test_fast, test_deep, selection, epsilon=epsilon)
            fast_threshold = select_f1_threshold(valid_y, valid_fast)
            empty_route = np.zeros(len(test_y), dtype=bool)
            metrics.append(metric_row(fast_name, seed, test_y, test_fast, fast_threshold, empty_route, {
                "routing_policy": "only", "requested_deep_budget": 0.0, "validation_f1": float("nan"),
                "fast_weight": 1.0, "fast_threshold": fast_threshold,
            }))
            cascade_name = f"{fast_name}->ProductionLevel2GATv2"
            metrics.append(metric_row(cascade_name, seed, test_y, final, selection.final_threshold, routed, {
                "routing_policy": "validation_calibrated", **asdict(selection),
            }))
            atomic_csv(prediction_root / f"{fast_name}__seed{seed}.csv", pd.DataFrame({
                "sample_id": test_ids,
                "label": test_y,
                "raw_fast_score": raw_test_fast,
                "calibrated_fast_score": test_fast,
                "raw_deep_score": raw_test_deep,
                "calibrated_deep_score": test_deep,
                "fusion_score": fused,
                "final_score": final,
                "deep_executed": routed.astype(int),
                "final_prediction": (final >= selection.final_threshold).astype(int),
            }))
            traces.append({
                "seed": seed,
                "fast_model": fast_name,
                "interface_case": closure_cfg["interface"]["case"],
                "fast_output_semantics": "probability",
                "deep_output_semantics": "probability",
                "calibration_input_space": "log_odds",
                "fusion_space": "calibrated_log_odds",
                "routing_score": "calibrated_fast_probability",
                "decision_score": "direct calibrated fast or routed calibrated fusion",
                "fast_calibrator": json.dumps(fast_calibrator.to_dict(), sort_keys=True),
                "deep_calibrator": json.dumps(deep_calibrator.to_dict(), sort_keys=True),
                **asdict(selection),
            })
            for split, labels, raw_f, cal_f, raw_d, cal_d in (
                ("validation", valid_y, raw_valid_fast, valid_fast, raw_valid_deep, valid_deep),
                ("test", test_y, raw_test_fast, test_fast, raw_test_deep, test_deep),
            ):
                for stage, values in (("fast_raw", raw_f), ("fast_calibrated", cal_f), ("deep_raw", raw_d), ("deep_calibrated", cal_d)):
                    for label in (0, 1):
                        selected = values[labels == label]
                        distributions.append({
                            "seed": seed, "model": fast_name, "split": split, "stage": stage, "label": label,
                            "N": len(selected), "mean": float(np.mean(selected)), "std": float(np.std(selected)),
                            "q01": float(np.quantile(selected, 0.01)), "q10": float(np.quantile(selected, 0.10)),
                            "q50": float(np.quantile(selected, 0.50)), "q90": float(np.quantile(selected, 0.90)),
                            "q99": float(np.quantile(selected, 0.99)),
                        })

            old = old_metrics[(old_metrics["model"] == cascade_name) & (old_metrics["routing_policy"] == "dual") & (old_metrics["seed"] == seed)]
            old_f1 = float(old.iloc[0]["f1"]) if len(old) else float("nan")
            new_f1 = float(binary_metrics(test_y, final, selection.final_threshold)["f1"])
            fast_f1 = float(binary_metrics(test_y, test_fast, fast_threshold)["f1"])
            acceptance.append({
                "seed": seed, "model": fast_name, "old_cascade_f1": old_f1, "calibrated_cascade_f1": new_f1,
                "calibrated_fast_f1": fast_f1, "delta_vs_old": new_f1 - old_f1,
                "delta_vs_fast": new_f1 - fast_f1, "deep_route_rate": float(routed.mean()),
            })
        print(f"calibrated seed={seed}", flush=True)

    metric_table = pd.DataFrame(metrics)
    acceptance_table = pd.DataFrame(acceptance)
    mean_delta = acceptance_table.groupby("model")["delta_vs_fast"].mean()
    if (mean_delta >= 0.0).all():
        gate = "PASS-A"
        interpretation = "calibrated cascades are retained; mean F1 does not degrade relative to the calibrated fast controls"
    elif (acceptance_table.groupby("model")["calibrated_cascade_f1"].mean() > acceptance_table.groupby("model")["old_cascade_f1"].mean()).all():
        gate = "PASS-B"
        interpretation = "collapse is corrected but cascade claims are restricted because fast-control parity is not reached"
    else:
        gate = "FAIL-C"
        interpretation = "tabular cascades are removed from the main claim set"
    atomic_csv(output / "cascade/calibrated_cascade_metrics.csv", metric_table)
    atomic_csv(output / "cascade/calibration_metrics.csv", pd.DataFrame(calibration))
    atomic_csv(output / "cascade/interface_trace.csv", pd.DataFrame(traces))
    atomic_csv(output / "cascade/score_distributions.csv", pd.DataFrame(distributions))
    atomic_csv(output / "cascade/acceptance_gate_by_seed.csv", acceptance_table)
    atomic_json(output / "cascade/acceptance_gate.json", {
        "gate": gate,
        "interpretation": interpretation,
        "mean_delta_vs_fast": mean_delta.to_dict(),
        "interface_case": closure_cfg["interface"]["case"],
    })
    atomic_json(output / "cascade/leakage_audit.json", {
        "status": "PASS",
        "selection_inputs": ["validation_labels", "validation_fast_scores", "validation_deep_scores"],
        "held_out_inputs": ["test_scores applied once after parameter freeze"],
        "forbidden_test_label_selection": True,
        "config_sha256": sha256_file(args.config),
        "base_config_sha256": sha256_file(base_path),
        "source_checkpoint_metadata_sha256": {
            str(seed): sha256_file(source / f"checkpoints/seed{seed}/metadata.json") for seed in closure_cfg["seeds"]
        },
    })
    print(json.dumps({"gate": gate, "rows": len(metric_table), "device": str(device), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
