"""Empirical multi-seed runner for the historical 8x10 DLG benchmark.

The runner is deliberately separate from the newer on-chain SCI orchestrator.
It reuses the historical loaders/model identities while adding leakage-explicit
thresholds, reproducibility metadata, checkpoint/resume, and failure isolation.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, remove_isolated_nodes, subgraph

from analysis.utils import dataset_metadata
from gog_fraud.evaluation.fraud_topology import compute_fraud_topology_metrics
from gog_fraud.evaluation.reproducibility import seed_everything
from gog_fraud.evaluation.threshold_protocol import evaluate_threshold_protocol
from gog_fraud.experiments.sci_round1 import (
    ROUND1_REQUIRED_COLUMNS, ResultStore, canonical_config_hash,
    experiment_key, export_architecture_metadata, summarize_multiseed,
)

log = logging.getLogger(__name__)


class PeakMemoryMonitor:
    def __init__(self, interval: float = 0.05) -> None:
        self.interval, self.peak_rss = interval, 0
        self._stop = threading.Event(); self._thread: threading.Thread | None = None

    def __enter__(self):
        process = psutil.Process(os.getpid())
        self.peak_rss = process.memory_info().rss
        def sample():
            while not self._stop.wait(self.interval):
                self.peak_rss = max(self.peak_rss, process.memory_info().rss)
        self._thread = threading.Thread(target=sample, daemon=True); self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


def _legacy_registries(data_root: str):
    # Importing does not execute main(). DATA_ROOT is overridden before loaders run.
    from scripts import benchmark_8x10_pipeline as legacy
    legacy.DATA_ROOT = data_root
    datasets = {
        "Elliptic": legacy.load_elliptic, "DGraphFin": legacy.load_dgraphfin,
        "Yelp": legacy.load_yelp, "Amazon": legacy.load_amazon,
        "BitcoinOTC": legacy.load_bitcoin_otc, "Flickr": legacy.load_flickr,
        "Reddit": legacy.load_reddit, "Cora": lambda: legacy.load_planetoid("Cora"),
        "CiteSeer": lambda: legacy.load_planetoid("CiteSeer"),
        "PubMed": lambda: legacy.load_planetoid("PubMed"),
    }
    models = {
        "DOMINANT": legacy.DOMINANT, "AnomalyDAE": legacy.AnomalyDAE,
        "CoLA": legacy.CoLA, "CONAD": legacy.CONAD, "GADNR": legacy.GADNR,
        "OCGNN": legacy.OCGNN, "DLG-Base": legacy.DLGBase, "DLG": legacy.DLG,
    }
    return datasets, models


def _select(names: list[str] | None, registry: dict[str, Any], kind: str) -> dict[str, Any]:
    if not names:
        return registry
    unknown = sorted(set(names).difference(registry))
    if unknown:
        raise ValueError(f"unknown {kind}: {unknown}")
    return {name: registry[name] for name in names}


def _limit_graph(data, max_nodes: int | None, seed: int):
    if not max_nodes or data.num_nodes <= max_nodes:
        return data
    y = data.y.detach().cpu().numpy().reshape(-1)
    eligible = np.arange(data.num_nodes)
    if hasattr(data, "eval_mask") and data.eval_mask is not None:
        eligible = np.flatnonzero(data.eval_mask.detach().cpu().numpy())
    rng = np.random.default_rng(seed)
    positive, negative = eligible[y[eligible] == 1], eligible[y[eligible] == 0]
    positive_budget = min(len(positive), max(2, int(round(max_nodes * len(positive) / max(1, len(eligible))))))
    negative_budget = min(len(negative), max_nodes - positive_budget)
    nodes = np.concatenate((rng.choice(positive, positive_budget, replace=False), rng.choice(negative, negative_budget, replace=False)))
    nodes.sort(); node_tensor = torch.as_tensor(nodes, dtype=torch.long)
    edge_index, _ = subgraph(node_tensor, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    edge_index, _, connected_mask = remove_isolated_nodes(edge_index, num_nodes=len(nodes))
    node_tensor = node_tensor[connected_mask]
    limited = data.clone(); limited.x = data.x[node_tensor].clone(); limited.y = data.y[node_tensor].clone()
    limited.edge_index = edge_index; limited.num_nodes = len(node_tensor)
    if hasattr(limited, "eval_mask"):
        limited.eval_mask = torch.ones(len(node_tensor), dtype=torch.bool)
    return limited


def _eligible_labels(data) -> tuple[np.ndarray, np.ndarray]:
    y = data.y.detach().cpu().numpy().reshape(-1).astype(np.int64)
    indices = np.arange(len(y))
    if hasattr(data, "eval_mask") and data.eval_mask is not None:
        mask = data.eval_mask.detach().cpu().numpy().astype(bool)
        indices, y = indices[mask], y[mask]
    valid = np.isin(y, (0, 1))
    return indices[valid], y[valid]


def _validation_test_indices(y: np.ndarray, seed: int, validation_ratio: float, test_ratio: float):
    if min(np.bincount(y, minlength=2)) < 2:
        raise ValueError("both classes need at least two samples for validation/test threshold evaluation")
    local = np.arange(len(y))
    val, test = train_test_split(
        local, test_size=test_ratio / (validation_ratio + test_ratio),
        random_state=seed, stratify=y,
    )
    return np.asarray(val), np.asarray(test)


def _fit_and_score(model_class, data, *, epochs: int, gpu: int, model_kwargs: dict[str, Any]):
    kwargs = {"epoch": epochs, "gpu": gpu, "batch_size": 0, "verbose": 0, **model_kwargs}
    try:
        model = model_class(num_neigh=-1, **kwargs)
    except TypeError:
        model = model_class(**kwargs)
    if torch.cuda.is_available() and gpu >= 0:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(gpu)
    with PeakMemoryMonitor() as memory:
        started = time.perf_counter(); model.fit(data); train_time = time.perf_counter() - started
        started = time.perf_counter(); scores = model.decision_function(data); inference_time = time.perf_counter() - started
    score = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
    peak_vram = torch.cuda.max_memory_allocated(gpu) / 2**20 if torch.cuda.is_available() and gpu >= 0 else 0.0
    return model, score.reshape(-1), train_time, inference_time, memory.peak_rss / 2**20, peak_vram


def _fit_and_score_partitioned(model_class, data, *, partition_size: int, epochs: int,
                               gpu: int, model_kwargs: dict[str, Any]):
    """Preserve the historical contiguous induced-subgraph strategy at scale."""
    if data.num_nodes <= partition_size:
        return _fit_and_score(model_class, data, epochs=epochs, gpu=gpu, model_kwargs=model_kwargs)
    scores, train_total, inference_total, ram_peak, vram_peak, last_model = [], 0.0, 0.0, 0.0, 0.0, None
    for start in range(0, data.num_nodes, partition_size):
        nodes = torch.arange(start, min(start + partition_size, data.num_nodes), dtype=torch.long)
        edge_index, _ = subgraph(nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(nodes))
        part = Data(x=data.x[nodes].clone(), y=data.y[nodes].clone(), edge_index=edge_index, num_nodes=len(nodes))
        if hasattr(data, "eval_mask") and data.eval_mask is not None:
            part.eval_mask = data.eval_mask[nodes].clone()
        last_model, part_score, train_time, inference_time, ram, vram = _fit_and_score(
            model_class, part, epochs=epochs, gpu=gpu, model_kwargs=model_kwargs)
        scores.append(part_score); train_total += train_time; inference_total += inference_time
        ram_peak, vram_peak = max(ram_peak, ram), max(vram_peak, vram)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return last_model, np.concatenate(scores), train_total, inference_total, ram_peak, vram_peak


def _resolve_gpu(requested: int) -> int:
    if requested < 0 or not torch.cuda.is_available():
        return -1
    try:
        probe = torch.zeros(1, device=f"cuda:{requested}")
        del probe
        torch.cuda.reset_peak_memory_stats(requested)
        torch.cuda.empty_cache()
        return requested
    except (RuntimeError, AssertionError):
        log.warning("CUDA device %s is advertised but unusable; falling back to CPU", requested)
        return -1


def _base_record(*, run_id: str, key: str, config_hash: str, dataset: str, model: str,
                 model_class, seed: int, data, split_type: str) -> dict[str, Any]:
    metadata = dataset_metadata(dataset)
    row = {column: None for column in ROUND1_REQUIRED_COLUMNS}
    row.update({
        "run_id": run_id, "experiment_key": key, "config_hash": config_hash,
        "dataset": dataset, **metadata, "model": model,
        "model_module": model_class.__module__, "model_class": model_class.__qualname__,
        "seed": seed, "variant": "l1_l2" if model == "DLG" else ("global_only" if model == "DLG-Base" else "baseline"),
        "split_type": split_type, "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges), "status": "failed",
    })
    return row


def run(config: dict[str, Any], *, output_root: Path, resume: bool, force: bool,
        dataset_filter: list[str] | None = None, model_filter: list[str] | None = None,
        seed_override: list[int] | None = None, max_nodes: int | None = None) -> int:
    evaluation = config.get("evaluation", {})
    seeds = [int(seed) for seed in (seed_override or evaluation.get("seeds", [42, 43, 44, 45, 46]))]
    data_root = str(config.get("data", {}).get("root", "/mnt/d/_Work/_data/DLG"))
    datasets, models = _legacy_registries(data_root)
    datasets = _select(dataset_filter or config.get("datasets"), datasets, "datasets")
    models = _select(model_filter or config.get("models"), models, "models")
    config_hash = canonical_config_hash(config); run_id = f"round1-{config_hash}-{uuid.uuid4().hex[:8]}"
    raw_path = output_root / "multiseed/raw_results.csv"; store = ResultStore.open(raw_path)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(exist_ok=True); (output_root / "topology").mkdir(exist_ok=True)
    manifest = {
        "run_id": run_id, "config_hash": config_hash, "seeds": seeds,
        "datasets": list(datasets), "models": list(models), "data_root": data_root,
        "split_strategy": "stratified_node_transductive",
        "threshold_protocol": "validation_selected_plus_explicit_test_oracle_and_topk",
        "dataset_seed": int(evaluation.get("dataset_seed", 42)), "failures": [],
    }
    gpu = _resolve_gpu(int(evaluation.get("gpu", 0 if torch.cuda.is_available() else -1)))
    manifest["requested_gpu"] = int(evaluation.get("gpu", 0 if torch.cuda.is_available() else -1))
    manifest["effective_gpu"] = gpu
    topology_rows: list[dict[str, Any]] = []
    topology_path = output_root / "topology/dataset_topology_metrics.csv"

    for dataset_name, loader in datasets.items():
        seed_everything(manifest["dataset_seed"])
        try:
            base_data = loader()
            if base_data is None:
                raise RuntimeError("loader returned None")
            base_data = _limit_graph(base_data, max_nodes or evaluation.get("max_nodes"), manifest["dataset_seed"])
            eligible, eligible_y = _eligible_labels(base_data)
            topology = compute_fraud_topology_metrics(base_data.edge_index, base_data.y, directed=bool(config.get("topology", {}).get("directed", True)))
            topology_rows.append({"dataset": dataset_name, **dataset_metadata(dataset_name), **topology.to_dict()})
            pd.DataFrame(topology_rows).to_csv(topology_path, index=False)
        except Exception as exc:
            manifest["failures"].append({"dataset": dataset_name, "stage": "load/topology", "type": type(exc).__name__, "message": str(exc)})
            log.exception("dataset failed: %s", dataset_name); continue

        for model_name, model_class in models.items():
            for seed in seeds:
                key = experiment_key(dataset=dataset_name, model=model_name, seed=seed,
                                     variant="paper", split_strategy=manifest["split_strategy"], config_hash=config_hash)
                if resume and not force and key in store.completed_keys:
                    log.info("[SKIP] %s/%s/seed=%d", dataset_name, model_name, seed); continue
                row = _base_record(run_id=run_id, key=key, config_hash=config_hash, dataset=dataset_name,
                                   model=model_name, model_class=model_class, seed=seed, data=base_data,
                                   split_type=manifest["split_strategy"])
                traceback_path = output_root / "logs" / f"{dataset_name}_{model_name}_{seed}.traceback.txt"
                try:
                    seed_everything(seed, deterministic=bool(evaluation.get("deterministic", True)))
                    data = base_data.clone()
                    model_kwargs = dict(config.get("model_kwargs", {}).get(model_name, {}))
                    partition_sizes = {"DGraphFin": 4096, "Yelp": 4096, "Reddit": 8192}
                    partition_size = int(evaluation.get("partition_sizes", {}).get(dataset_name, partition_sizes.get(dataset_name, 16384)))
                    model, scores, train_time, inference_time, peak_ram, peak_vram = _fit_and_score_partitioned(
                        model_class, data, partition_size=partition_size, epochs=int(evaluation.get("epochs", 50)),
                        gpu=gpu, model_kwargs=model_kwargs,
                    )
                    val_local, test_local = _validation_test_indices(
                        eligible_y, seed, float(evaluation.get("validation_ratio", 0.2)), float(evaluation.get("test_ratio", 0.2)))
                    selected_scores = scores[eligible]
                    threshold = evaluate_threshold_protocol(
                        eligible_y[val_local], selected_scores[val_local], eligible_y[test_local], selected_scores[test_local])
                    test_y, test_score = eligible_y[test_local], selected_scores[test_local]
                    row.update(threshold.to_dict())
                    row.update({
                        "roc_auc": float(roc_auc_score(test_y, test_score)),
                        "pr_auc": float(average_precision_score(test_y, test_score)),
                        "train_time_sec": train_time, "inference_time_sec": inference_time,
                        "peak_ram_mb": peak_ram, "peak_vram_mb": peak_vram,
                        "num_nodes": int(len(eligible_y)),
                        "num_positive": int((eligible_y == 1).sum()),
                        "num_negative": int((eligible_y == 0).sum()),
                        "positive_ratio": float(eligible_y.mean()), "status": "success",
                    })
                    export_architecture_metadata(
                        output_root / f"manifests/architectures/{model_name}_{seed}.json",
                        model=model, config={"epochs": evaluation.get("epochs", 50), **model_kwargs},
                        paper_name=model_name, variant=row["variant"],
                    )
                    log.info("[OK] %s/%s/seed=%d PR=%.4f", dataset_name, model_name, seed, row["pr_auc"])
                except Exception as exc:
                    traceback_path.parent.mkdir(parents=True, exist_ok=True)
                    traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
                    message = str(exc); lower = message.lower()
                    row.update({
                        "status": "oom" if "out of memory" in lower or "cannot allocate" in lower else "failed",
                        "error_type": type(exc).__name__, "error_message": message,
                        "traceback_path": str(traceback_path),
                        "num_nodes": int(len(eligible_y)),
                        "num_positive": int((eligible_y == 1).sum()), "num_negative": int((eligible_y == 0).sum()),
                        "positive_ratio": float(eligible_y.mean()),
                    })
                    manifest["failures"].append({"dataset": dataset_name, "model": model_name, "seed": seed, "type": type(exc).__name__, "message": message})
                    log.exception("[FAILED] %s/%s/seed=%d", dataset_name, model_name, seed)
                store.append(row)
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

    if store.rows:
        summary = summarize_multiseed(pd.DataFrame(store.rows))
        summary_path = output_root / "multiseed/summary_results.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True); summary.to_csv(summary_path, index=False)
    manifest["status"] = "completed_with_failures" if manifest["failures"] else "success"
    (output_root / f"manifests/{run_id}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if store.rows else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output-root")
    parser.add_argument("--datasets", nargs="+"); parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int); parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    output_root = Path(args.output_root or config.get("experiment", {}).get("output_root", "outputs/sci"))
    return run(config, output_root=output_root, resume=args.resume, force=args.force,
               dataset_filter=args.datasets, model_filter=args.models, seed_override=args.seeds, max_nodes=args.max_nodes)


if __name__ == "__main__":
    raise SystemExit(main())
