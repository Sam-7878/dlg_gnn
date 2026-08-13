"""Empirical DLG local/global/fusion ablation for SCI round 1."""
from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from gog_fraud.evaluation.reproducibility import seed_everything
from gog_fraud.evaluation.threshold_protocol import best_f1_threshold, evaluate_threshold_protocol
from gog_fraud.pipelines.run_sci_round1_benchmark import (
    _eligible_labels, _fit_and_score, _legacy_registries, _limit_graph,
    _resolve_gpu, _validation_test_indices,
)

log = logging.getLogger(__name__)


def _validation_scale(validation: np.ndarray, all_scores: np.ndarray) -> np.ndarray:
    low, high = float(np.min(validation)), float(np.max(validation))
    if high <= low:
        return np.zeros_like(all_scores, dtype=float)
    return np.clip((all_scores - low) / (high - low), 0.0, 1.0)


def _metric_row(dataset: str, seed: int, variant: str, y: np.ndarray, score: np.ndarray,
                val: np.ndarray, test: np.ndarray, *, train_time: float,
                inference_time: float, peak_ram: float, peak_vram: float,
                extra: dict | None = None) -> dict:
    threshold = evaluate_threshold_protocol(y[val], score[val], y[test], score[test])
    row = {
        "dataset": dataset, "seed": seed, "variant": variant,
        "roc_auc": float(roc_auc_score(y[test], score[test])),
        "pr_auc": float(average_precision_score(y[test], score[test])),
        **threshold.to_dict(), "train_time_sec": train_time,
        "inference_time_sec": inference_time, "peak_ram_mb": peak_ram,
        "peak_vram_mb": peak_vram, "status": "success", "error_type": None,
        "error_message": None,
    }
    row.update(extra or {}); return row


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["roc_auc", "pr_auc", "validation_f1", "train_time_sec", "peak_ram_mb", "peak_vram_mb"]
    summary = raw.loc[raw.status.eq("success")].groupby(["dataset", "variant"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["dataset", "variant", *[f"{metric}_{stat}" for metric in metrics for stat in ("mean", "std")]]
    for reference, suffix in (("l1_only", "vs_l1"), ("l1_l2", "vs_l1_l2")):
        ref = summary.loc[summary.variant.eq(reference)].set_index("dataset")
        for metric in ("roc_auc", "pr_auc", "validation_f1"):
            summary[f"delta_{metric}_{suffix}"] = summary[metric + "_mean"] - summary.dataset.map(ref[metric + "_mean"])
    return summary


def run(config: dict, output_root: Path, *, datasets: list[str] | None = None,
        seeds: list[int] | None = None, max_nodes: int | None = None) -> int:
    evaluation = config.get("evaluation", {})
    seed_values = [int(value) for value in (seeds or evaluation.get("seeds", [42, 43, 44, 45, 46]))]
    dataset_registry, model_registry = _legacy_registries(config.get("data", {}).get("root", "/mnt/d/_Work/_data/DLG"))
    names = datasets or config.get("datasets") or list(dataset_registry)
    gpu = _resolve_gpu(int(evaluation.get("gpu", 0 if torch.cuda.is_available() else -1)))
    rows, failures = [], []
    weights = [float(value) for value in config.get("fusion", {}).get("weighted_l1", [0.2, 0.4, 0.5, 0.6, 0.8])]
    for dataset in names:
        seed_everything(int(evaluation.get("dataset_seed", 42)))
        base = _limit_graph(dataset_registry[dataset](), max_nodes or evaluation.get("max_nodes"), int(evaluation.get("dataset_seed", 42)))
        eligible, y = _eligible_labels(base)
        for seed in seed_values:
            val, test = _validation_test_indices(y, seed, float(evaluation.get("validation_ratio", .2)), float(evaluation.get("test_ratio", .2)))
            seed_everything(seed)
            try:
                global_data = base.clone()
                _, global_score, train_time, inference_time, ram, vram = _fit_and_score(
                    model_registry["DLG-Base"], global_data, epochs=int(evaluation.get("epochs", 50)), gpu=gpu, model_kwargs={})
                rows.append(_metric_row(dataset, seed, "global_only", y, global_score[eligible], val, test,
                                        train_time=train_time, inference_time=inference_time, peak_ram=ram, peak_vram=vram))
            except Exception as exc:
                failures.append({"dataset": dataset, "seed": seed, "stage": "global_only", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
                rows.append({"dataset": dataset, "seed": seed, "variant": "global_only", "status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)})
            seed_everything(seed)
            try:
                layered_data = base.clone()
                _, layered_score, train_time, inference_time, ram, vram = _fit_and_score(
                    model_registry["DLG"], layered_data, epochs=int(evaluation.get("epochs", 50)), gpu=gpu, model_kwargs={})
                local_score = layered_data.dlg_l1_score.detach().cpu().numpy().reshape(-1)
                local, layered = local_score[eligible], layered_score[eligible]
                rows.append(_metric_row(dataset, seed, "l1_only", y, local, val, test,
                                        train_time=train_time, inference_time=0.0, peak_ram=ram, peak_vram=vram,
                                        extra={"shared_training_with": "l1_l2"}))
                rows.append(_metric_row(dataset, seed, "l1_l2", y, layered, val, test,
                                        train_time=train_time, inference_time=inference_time, peak_ram=ram, peak_vram=vram))
                local_scaled = _validation_scale(local[val], local)
                layered_scaled = _validation_scale(layered[val], layered)
                selected_weight, selected_validation_f1 = None, -1.0
                for weight in weights:
                    candidate = weight * local_scaled + (1.0 - weight) * layered_scaled
                    candidate_f1, _ = best_f1_threshold(y[val], candidate[val])
                    if candidate_f1 > selected_validation_f1:
                        selected_weight, selected_validation_f1 = weight, candidate_f1
                fused = selected_weight * local_scaled + (1.0 - selected_weight) * layered_scaled
                rows.append(_metric_row(dataset, seed, "full", y, fused, val, test,
                                        train_time=train_time, inference_time=inference_time, peak_ram=ram, peak_vram=vram,
                                        extra={"fusion": "weighted_sum", "selected_l1_weight": selected_weight,
                                               "weight_selection": "validation_best_f1"}))
            except Exception as exc:
                failures.append({"dataset": dataset, "seed": seed, "stage": "layered", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
                for variant in ("l1_only", "l1_l2", "full"):
                    rows.append({"dataset": dataset, "seed": seed, "variant": variant, "status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)})
    raw = pd.DataFrame(rows); target = output_root / "ablation"; target.mkdir(parents=True, exist_ok=True)
    raw.to_csv(target / "ablation_raw.csv", index=False)
    summary = _summarize(raw); summary.to_csv(target / "ablation_summary.csv", index=False)
    (target / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    return 0 if raw.status.eq("success").any() else 2


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-root", default="outputs/sci")
    parser.add_argument("--datasets", nargs="+"); parser.add_argument("--seeds", nargs="+", type=int); parser.add_argument("--max-nodes", type=int)
    args = parser.parse_args(); logging.basicConfig(level=logging.INFO); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    return run(config, Path(args.output_root), datasets=args.datasets, seeds=args.seeds, max_nodes=args.max_nodes)


if __name__ == "__main__": raise SystemExit(main())

