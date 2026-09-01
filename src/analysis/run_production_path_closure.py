"""Train and evaluate the frozen production-path controls for SCI-v3 submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader

from gog_fraud.production.closure import (
    ambiguity_cutoff,
    build_graph_cache,
    fuse_scores,
    infer_level1,
    infer_level2,
    relation_data,
    save_seed_bundle,
    train_level1,
    train_level2,
    train_tabular,
)
from validation.sci_v3_final_common import atomic_csv, atomic_json, binary_metrics, select_f1_threshold, sha256_file


def support(labels: np.ndarray) -> dict[str, Any]:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    return {
        "N": int(len(labels)), "N_positive": positives, "N_negative": negatives,
        "metric_defined": bool(positives > 0 and negatives > 0),
        "undefined_reason": "" if positives > 0 and negatives > 0 else "single_class_target",
    }


def conservative_margin(scores: np.ndarray, labels: np.ndarray, threshold: float, alpha: float, maximum: float) -> float:
    """Validation-constrained routing margin; not a distribution-free RCPS guarantee."""
    candidates = np.unique(np.r_[0.0, np.quantile(np.abs(scores - threshold), np.linspace(0, 1, 101)), maximum])
    accepted = 0.0
    for margin in candidates:
        direct = np.abs(scores - threshold) > margin
        positive = direct & (labels == 1)
        if positive.sum() and float(((scores[positive] < threshold).sum()) / positive.sum()) <= alpha:
            accepted = float(margin)
    return min(accepted, maximum)


def row(model: str, route: str, seed: int, labels: np.ndarray, scores: np.ndarray, threshold: float, deep: np.ndarray) -> dict[str, Any]:
    metrics = binary_metrics(labels, scores, threshold)
    return {
        "model": model, "routing_policy": route, "seed": seed, "threshold": threshold,
        "deep_route_rate": float(deep.mean()), "direct_exit_rate": float(1.0 - deep.mean()),
        "measurement_type": "measured_prediction", **support(labels), **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sci_v3_submission/production_closure.yaml"))
    parser.add_argument("--identity-only", action="store_true", help="refresh identity metadata without retraining deterministic checkpoints")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path("results/sci_v3_submission")
    output.mkdir(parents=True, exist_ok=True)
    if args.identity_only:
        for seed in map(int, cfg["seeds"]):
            path = output / f"checkpoints/seed{seed}/metadata.json"
            metadata = json.loads(path.read_text(encoding="utf-8")); metadata["method_identity"] = cfg["method_identity"]
            metadata["config_sha256"] = sha256_file(args.config); atomic_json(path, metadata)
        atomic_json(output / "method_identity.json", {**cfg["method_identity"], "decision": "Path A",
            "production_backbone_seeds": list(map(int, cfg["seeds"])), "mixed_terminology_permitted": False,
            "config": str(args.config), "measurement_type": "measured_prediction"})
        print(json.dumps({"status": "identity_refreshed", "decision": "Path A"})); return
    dataset_root = Path(cfg["dataset_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bounded = cfg["bounded_graph"]
    cache = build_graph_cache(dataset_root, output / "cache/bounded_graphs.pt", int(bounded["max_edges"]), int(bounded["max_nodes"]))
    graphs = cache["graphs"]
    batch_size = int(cfg["level1"]["batch_size"])
    records: list[dict[str, Any]] = []
    prediction_root = output / "baselines/predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)

    for seed in map(int, cfg["seeds"]):
        level1, level1_meta = train_level1(graphs["train"], graphs["validation"], cfg["level1"], seed, device)
        inferred = {}
        for split in ("train", "validation", "test"):
            inferred[split] = infer_level1(level1, DataLoader(graphs[split], batch_size=batch_size, shuffle=False), device)
        train_score, train_y, train_embedding = inferred["train"]
        valid_score, valid_y, valid_embedding = inferred["validation"]
        test_score, test_y, test_embedding = inferred["test"]
        level1_threshold = select_f1_threshold(valid_y, valid_score)

        train_relation = relation_data(train_embedding, train_score, train_y, None, None, None, int(cfg["level2"]["knn_k"]))
        level2, level2_meta = train_level2(train_relation, train_embedding.shape[1] + 1, cfg["level2"], seed, device)
        valid_relation = relation_data(train_embedding, train_score, train_y, valid_embedding, valid_score, valid_y, int(cfg["level2"]["knn_k"]))
        test_relation = relation_data(train_embedding, train_score, train_y, test_embedding, test_score, test_y, int(cfg["level2"]["knn_k"]))
        valid_deep = infer_level2(level2, valid_relation, len(train_y), device)
        test_deep = infer_level2(level2, test_relation, len(train_y), device)
        valid_fusion = fuse_scores(valid_score, valid_deep, cfg["fusion"])
        test_fusion = fuse_scores(test_score, test_deep, cfg["fusion"])

        tabular, tabular_scores = train_tabular(dataset_root, seed)
        model_scores = {
            "XGBoostFastTriage": tabular_scores["XGBoostFastTriage"][:2],
            "LightGBMFastTriage": tabular_scores["LightGBMFastTriage"][:2],
            "ProductionLevel1GIN": (valid_score, test_score),
        }
        thresholds: dict[str, float] = {}
        cutoffs: dict[str, dict[str, float]] = {}
        for name, (validation_score, testing_score) in model_scores.items():
            threshold = select_f1_threshold(valid_y, validation_score)
            dual = ambiguity_cutoff(validation_score, threshold, float(cfg["routing"]["deep_budget"]))
            risk = conservative_margin(validation_score, valid_y, threshold, 0.10, dual)
            thresholds[name] = threshold
            cutoffs[name] = {"dual": dual, "risk_controlled": risk}
            none = np.zeros(len(test_y), dtype=bool)
            records.append(row(name, "only", seed, test_y, testing_score, threshold, none))
            for policy, cutoff in cutoffs[name].items():
                routed = np.abs(testing_score - threshold) <= cutoff
                cascade = np.where(routed, test_fusion, testing_score)
                records.append(row(f"{name}->ProductionLevel2GATv2", policy, seed, test_y, cascade, threshold, routed))
                atomic_csv(prediction_root / f"{name}__{policy}__seed{seed}.csv", pd.DataFrame({
                    "sample_id": [item["sample_id"] for item in cache["metadata"]["test"]],
                    "label": test_y, "fast_score": testing_score, "deep_score": test_deep,
                    "fusion_score": test_fusion, "final_score": cascade, "deep_executed": routed,
                }))

        full = np.ones(len(test_y), dtype=bool)
        records.append(row("ProductionLevel1GIN->ProductionLevel2GATv2", "no_routing", seed, test_y, test_fusion, level1_threshold, full))
        metadata = {
            "seed": seed, "device": str(device), "level1": level1_meta, "level2": level2_meta,
            "method_identity": cfg["method_identity"], "thresholds": thresholds, "cutoffs": cutoffs,
            "risk_control_interpretation": "validation-constrained empirical routing rule; not distribution-free RCPS",
            "split_files": {chain: sha256_file(dataset_root / f"splits/{chain}_holdout_v2.json") for chain in ("ethereum", "bsc", "polygon")},
            "config_sha256": sha256_file(args.config), "test_support": support(test_y),
        }
        save_seed_bundle(output / f"checkpoints/seed{seed}", level1, level2, metadata, tabular)
        print(f"completed seed={seed}", flush=True)

    table = pd.DataFrame(records)
    atomic_csv(output / "baselines/production_backbone_metrics.csv", table)
    atomic_json(output / "method_identity.json", {
        **cfg["method_identity"], "decision": "Path A", "production_backbone_seeds": list(map(int, cfg["seeds"])),
        "mixed_terminology_permitted": False, "config": str(args.config), "measurement_type": "measured_prediction",
    })
    print(json.dumps({"rows": len(table), "device": str(device), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
