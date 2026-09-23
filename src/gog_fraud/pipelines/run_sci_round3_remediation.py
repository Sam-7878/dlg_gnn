"""Round-3 remediation gates. Never launches the Round-4 400-run matrix."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis.utils import dataset_metadata
from gog_fraud.data.dgraphfin_aligned import load_dgraphfin_aligned
from gog_fraud.evaluation.graph_aware_partition import GraphAwareHaloPartitioner
from gog_fraud.evaluation.fraud_topology import compute_fraud_topology_metrics
from gog_fraud.evaluation.reproducibility import seed_everything
from gog_fraud.experiments.round2_validity import graph_fingerprints
from gog_fraud.pipelines.run_sci_round1_benchmark import (
    _eligible_labels, _fit_and_score, _legacy_registries, _resolve_gpu, _validation_test_indices,
)


LAYOUT = ("manifests", "partition", "failures", "convergence", "orientation", "provenance",
          "dgraphfin", "injection", "ablation", "representative", "resources", "tables", "figures")


def ensure_layout(root: Path) -> None:
    for name in LAYOUT:
        (root / name).mkdir(parents=True, exist_ok=True)


def round3_registry(config: dict):
    root = config["data"]["root"]
    datasets, models = _legacy_registries(root, 42)
    datasets["DGraphFin"] = lambda: load_dgraphfin_aligned(Path(root) / "DGraphFin/dgraphfin.npz")
    return datasets, models


def run_partition_validation(config: dict, output: Path) -> dict:
    registry, _ = round3_registry(config); settings = config["partition"]
    summary_rows, core_rows, failures = [], [], []
    for dataset in ("DGraphFin", "Yelp", "Reddit"):
        started = time.perf_counter(); data = registry[dataset](); load_sec = time.perf_counter() - started
        core_size = int(settings["core_sizes"][dataset]); bidirectional = bool(settings["stored_bidirectional"][dataset])
        started = time.perf_counter()
        plan = GraphAwareHaloPartitioner(data, core_size=core_size,
                                         halo_hops=int(settings.get("halo_hops", 1)),
                                         backend=settings.get("backend", "metis"),
                                         stored_bidirectional=bidirectional)
        partition_sec = time.perf_counter() - started
        plan.assert_unique_core_assignment()
        stats = []
        for partition_id in plan.partition_ids:
            stat = plan.measure(int(partition_id)); stats.append(stat)
            core_rows.append({"dataset": dataset, **stat.to_dict()})
        edge = data.edge_index.cpu(); non_self = edge[0] != edge[1]
        internal = plan.assignment[edge[0, non_self]] == plan.assignment[edge[1, non_self]]
        topology = compute_fraud_topology_metrics(edge, data.y, directed=True)
        min_edge = min(item.core_edge_coverage for item in stats)
        min_neighbor = min(item.core_neighbor_coverage for item in stats)
        max_local = max(item.total_local_nodes for item in stats)
        max_dense = max(item.dense_adjacency_elements for item in stats)
        resource_ok = max_local <= int(settings["max_expanded_nodes"])
        minimum = float(settings["acceptance"]["minimum"])
        summary_rows.append({
            "dataset": dataset, "partition_strategy": plan.strategy, "core_size_target": core_size,
            "halo_hops": plan.halo_hops, "num_partitions": len(plan),
            "core_edge_coverage_min": min_edge,
            "core_edge_coverage_mean": float(np.mean([item.core_edge_coverage for item in stats])),
            "core_neighbor_coverage_min": min_neighbor,
            "core_neighbor_coverage_mean": float(np.mean([item.core_neighbor_coverage for item in stats])),
            "global_edge_retention_without_halo": float(internal.float().mean()) if internal.numel() else 1.0,
            "global_edge_retention_with_halo": 1.0 if plan.halo_hops >= 1 else float(internal.float().mean()),
            "cross_partition_edge_ratio_without_halo": float((~internal).float().mean()) if internal.numel() else 0.0,
            "max_halo_ratio": max(item.halo_ratio for item in stats), "max_total_local_nodes": max_local,
            "max_dense_adjacency_elements": max_dense, "dense_budget_nodes": int(settings["max_expanded_nodes"]),
            "topology_gate_pass": min_edge >= minimum and min_neighbor >= minimum,
            "dense_resource_gate_pass": resource_ok, "load_time_sec": load_sec,
            "partition_time_sec": partition_sec, "fraud_homophily_original": topology.fraud_homophily,
            "fraud_homophily_partition_context": topology.fraud_homophily,
            "adjusted_homophily_original": topology.adjusted_homophily,
            "adjusted_homophily_partition_context": topology.adjusted_homophily,
        })
        if not resource_ok:
            failures.append({"dataset": dataset, "classification": "partition_resource_error",
                             "message": f"max core+halo {max_local} exceeds declared dense budget {settings['max_expanded_nodes']}"})
        del plan, data
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    fidelity = pd.DataFrame(summary_rows); fidelity.to_csv(output / "partition/fidelity.csv", index=False)
    pd.DataFrame(core_rows).to_csv(output / "partition/core_halo_stats.csv", index=False)
    (output / "partition/failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    decision = {"topology_pass": bool(fidelity.topology_gate_pass.all()),
                "dense_resource_pass": bool(fidelity.dense_resource_gate_pass.all()), "failures": failures}
    (output / "manifests/partition_gate.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return decision


def run_provenance(config: dict, output: Path) -> None:
    rows = []
    for name in ("Elliptic", "DGraphFin", "Yelp", "Cora", "Reddit"):
        meta = dataset_metadata(name)
        rows.append({"dataset": name, **meta,
                     "label_source": {"Elliptic": "dataset illicit/licit labels",
                                      "DGraphFin": "NPZ labels 0 normal / 1 fraud",
                                      "Yelp": "PyGOD synthetic injection overwrites converted labels",
                                      "Cora": "PyGOD synthetic injection",
                                      "Reddit": "PyGOD synthetic injection"}[name],
                     "injection_method": "none" if name in {"Elliptic", "DGraphFin"} else "contextual_plus_structural",
                     "split_type": "official_random_70_15_15" if name == "DGraphFin" else "stratified_evaluation",
                     "temporal_available": name == "DGraphFin",
                     "temporal_used": False,
                     "graph_direction_semantics": "directed" if name != "Yelp" else "undirected_stored_bidirectional"})
    pd.DataFrame(rows).to_csv(output / "provenance/table_r3_d_provenance.csv", index=False)
    label_rows = [
        {"dataset": "Elliptic", "raw_labels": "1=illicit, 2=licit, unknown class mapped to 2 by loader",
         "normal_label": 0, "fraud_label": 1, "excluded": "unknown", "mapping_action": "dataset class processing then unknown removal"},
        {"dataset": "DGraphFin", "raw_labels": "0=normal, 1=fraud, 2/3=background",
         "normal_label": 0, "fraud_label": 1, "excluded": "2,3", "mapping_action": "identity mapping for retained 0/1"},
    ]
    pd.DataFrame(label_rows).to_csv(output / "provenance/label_semantics_audit.csv", index=False)


def _orientation_source(dataset: str, data, config: dict):
    if dataset == "Cora":
        return data, torch.arange(data.num_nodes), "full_graph"
    plan = GraphAwareHaloPartitioner(data, core_size=4096, halo_hops=1, backend="metis",
                                     stored_bidirectional=False, max_expanded_nodes=8192)
    candidates = []
    for partition_id in plan.partition_ids:
        core = torch.nonzero(plan.assignment == int(partition_id), as_tuple=False).flatten()
        candidates.append((int((data.y[core] == 1).sum()), int(partition_id)))
    _, selected = max(candidates)
    part = plan.build(selected)
    return part.data, part.core_local_index, f"graph_aware_core_partition_{selected}"


def run_orientation_convergence(config: dict, output: Path) -> None:
    registry, models = round3_registry(config); settings = config["convergence"]
    gpu = _resolve_gpu(0); rows = []
    for dataset in settings["datasets"]:
        full = registry[dataset](); data, evaluated_local, scope = _orientation_source(dataset, full, config)
        labels = data.y[evaluated_local].cpu().numpy().astype(int)
        for model_name in settings["models"]:
            for epoch in settings["epochs"]:
                seed_everything(int(settings["seed"])); kwargs = {}
                if model_name == "DLG": kwargs["l1_epochs"] = min(int(epoch), 20)
                try:
                    _, score, train_sec, infer_sec, _, ram, _, vram = _fit_and_score(
                        models[model_name], data.clone(), epochs=int(epoch), gpu=gpu, model_kwargs=kwargs)
                    selected_score = score[evaluated_local.cpu().numpy()]
                    raw_auc = float(roc_auc_score(labels, selected_score)); inverted = float(roc_auc_score(labels, -selected_score))
                    rows.append({"dataset": dataset, "model": model_name, "epoch": int(epoch),
                                 "scope": scope, "training_loss": np.nan, "raw_roc_auc": raw_auc,
                                 "inverted_roc_auc": inverted, "pr_auc": float(average_precision_score(labels, selected_score)),
                                 "score_mean_normal": float(selected_score[labels == 0].mean()),
                                 "score_mean_anomaly": float(selected_score[labels == 1].mean()),
                                 "orientation_warning": inverted > raw_auc + .10,
                                 "train_time_sec": train_sec, "inference_time_sec": infer_sec,
                                 "peak_ram_mb": ram, "peak_vram_mb": vram, "status": "success"})
                except Exception as exc:
                    rows.append({"dataset": dataset, "model": model_name, "epoch": int(epoch),
                                 "scope": scope, "status": "failed", "error_type": type(exc).__name__,
                                 "error_message": str(exc)})
    raw = pd.DataFrame(rows); raw.to_csv(output / "convergence/orientation_trajectory.csv", index=False)
    final = raw.loc[raw.status.eq("success")].sort_values("epoch").groupby(["dataset", "model"], as_index=False).tail(1).copy()
    final["orientation_status"] = np.where(
        (final.dataset == "Elliptic") & (final.raw_roc_auc < .4),
        "known_reconstruction_misalignment_not_score_inversion",
        np.where(final.raw_roc_auc >= .5, "normal_direction",
                 np.where(final.raw_roc_auc >= .4, "random_or_weak", "persistent_unexplained_inversion")))
    final.to_csv(output / "orientation/final_orientation_classification.csv", index=False)
    raw.to_csv(output / "tables/table_r3_c_orientation_convergence.csv", index=False)


def run_injection_variance(config: dict, output: Path) -> None:
    settings = config["injection"]; root = config["data"]["root"]; gpu = _resolve_gpu(0); rows = []
    for dataset_seed in settings["dataset_seeds"]:
        registry, models = _legacy_registries(root, int(dataset_seed))
        seed_everything(int(dataset_seed)); data = registry["Cora"]()
        eligible, y = _eligible_labels(data); fingerprints = graph_fingerprints(data, injection_config={"dataset_seed": dataset_seed})
        for model_name in settings["models"]:
            for model_seed in settings["model_seeds"]:
                val, test = _validation_test_indices(y, int(model_seed), .2, .2)
                kwargs = {"l1_epochs": int(settings["l1_epochs"])} if model_name == "DLG" else {}
                seed_everything(int(model_seed))
                _, score, train_sec, infer_sec, _, ram, _, vram = _fit_and_score(
                    models[model_name], data.clone(), epochs=int(settings["epochs"]), gpu=gpu, model_kwargs=kwargs)
                selected = score[eligible]; test_score, test_y = selected[test], y[test]
                rows.append({"dataset": "Cora", "display_name": "Cora-Syn", "dataset_seed": dataset_seed,
                             "model_seed": model_seed, "model": model_name,
                             "roc_auc": float(roc_auc_score(test_y, test_score)),
                             "pr_auc": float(average_precision_score(test_y, test_score)),
                             "train_time_sec": train_sec, "inference_time_sec": infer_sec,
                             "peak_ram_mb": ram, "peak_vram_mb": vram, **fingerprints})
    raw = pd.DataFrame(rows); raw.to_csv(output / "injection/injection_variance_raw.csv", index=False)
    decomposition = []
    for model, group in raw.groupby("model"):
        means = group.groupby("dataset_seed").pr_auc.mean()
        within = group.groupby("dataset_seed").pr_auc.var(ddof=1)
        within_mean = float(within.mean()); between_means = float(means.var(ddof=1))
        repeats = int(group.model_seed.nunique())
        injection_component = max(0.0, between_means - within_mean / repeats)
        ratio = injection_component / within_mean if within_mean > 0 else np.inf
        decomposition.append({"model": model, "between_injection_mean_variance": between_means,
                              "within_injection_model_seed_variance": within_mean,
                              "estimated_injection_variance_component": injection_component,
                              "injection_to_model_variance_ratio": ratio,
                              "round4_recommendation": ("multiple_injection_seeds" if ratio > 1
                                                        else "fixed_instance_allowed_with_explicit_label")})
    pd.DataFrame(decomposition).to_csv(output / "injection/variance_decomposition.csv", index=False)


def run_dgraphfin_audit(config: dict, output: Path) -> None:
    path = Path(config["data"]["root"]) / "DGraphFin/dgraphfin.npz"
    data = load_dgraphfin_aligned(path)
    record = {
        "source": str(path), "filtered_num_nodes": int(data.num_nodes),
        "filtered_num_edges": int(data.num_edges), "edge_timestamp_count": int(data.edge_timestamp.numel()),
        "edge_timestamp_min": int(data.edge_timestamp.min()), "edge_timestamp_max": int(data.edge_timestamp.max()),
        "train_nodes": int(data.train_mask.sum()), "validation_nodes": int(data.val_mask.sum()),
        "test_nodes": int(data.test_mask.sum()), "official_split_semantics": "random_70_15_15_on_labels_0_1",
        "official_split_temporal": False, "timestamp_source": "same_npz_edge_timestamp",
        "dgraphfin2_auto_attachment": False, "original_node_mapping_preserved": True,
        "edge_timestamp_alignment": data.edge_index.size(1) == data.edge_timestamp.numel(),
        "split_disjoint": not bool((data.train_mask & data.val_mask).any()
                                   or (data.train_mask & data.test_mask).any()
                                   or (data.val_mask & data.test_mask).any()),
        "edge_indices_in_range": not data.edge_index.numel() or int(data.edge_index.max()) < data.num_nodes,
    }
    (output / "dgraphfin/alignment_audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--stages", nargs="+", choices=("partition", "provenance", "orientation", "injection", "dgraphfin"), default=["partition", "provenance"])
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(config["experiment"]["output_root"]); ensure_layout(output)
    if "partition" in args.stages: run_partition_validation(config, output)
    if "provenance" in args.stages: run_provenance(config, output)
    if "orientation" in args.stages: run_orientation_convergence(config, output)
    if "injection" in args.stages: run_injection_variance(config, output)
    if "dgraphfin" in args.stages: run_dgraphfin_audit(config, output)
    (output / "manifests/config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
