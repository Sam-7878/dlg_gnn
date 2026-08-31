#!/usr/bin/env python3
"""Scope-explicit profiler for the actual ContractDLG SCI evidence path."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from gog_fraud.pipelines.run_round4_experiments import SciV2Records, _data, _fit_dlg, _normalize
from validation.sci_v3_final_common import atomic_csv, atomic_json


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "repetitions": len(array),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "std_ms": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def measure(fn: Callable[[], object], device: torch.device, warmup: int, repetitions: int) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    values = []
    for _ in range(repetitions):
        started = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    return summarize(values)


def run(dataset_root: Path, output_dir: Path, chain: str, seed: int, epochs: int, warmup: int, repetitions: int) -> pd.DataFrame:
    dataset = SciV2Records(dataset_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_ids, test_ids = dataset.ids(chain, "train"), dataset.ids(chain, "test")
    train_x, train_y = dataset.arrays(train_ids)
    test_x, _ = dataset.arrays(test_ids)
    train_x, test_x = _normalize(train_x, test_x)
    model, _ = _fit_dlg(train_x, train_y, variant="DLG-Full-Fusion", seed=seed, epochs=epochs, device=device)
    model.eval()
    graph = _data(train_x, test_x).to(device)
    offset = len(train_x)
    x, edge_index = graph.x, graph.edge_index
    with torch.no_grad():
        local_cached = model.local(x)
        relation_cached = torch.relu(model.conv1(local_cached, edge_index))
        relation_cached = torch.relu(model.conv2(relation_cached, edge_index))

    def run_end_to_end() -> torch.Tensor:
        prepared = _data(train_x, test_x).to(device)
        return model(prepared.x, prepared.edge_index)[offset:]

    functions: dict[str, tuple[str, Callable[[], object]]] = {
        "contract_summary_feature_preparation": ("end_to_end", lambda: _normalize(train_x, test_x)),
        "local_relation_graph_construction": ("end_to_end", lambda: _data(train_x, test_x)),
        "Level-1 contract-summary encoder": ("model_only", lambda: model.local(x)),
        "Level-1 fraud head": ("model_only", lambda: model.local_head(local_cached)[offset:]),
        "MC stochastic Level-1 path T=8": (
            "model_only",
            lambda: [model.local_head(model.local(x))[offset:] for _ in range(8)],
        ),
        "routing decision": (
            "end_to_end",
            lambda: ((torch.sigmoid(model.local_head(local_cached)[offset:]) < 0.35) | (torch.sigmoid(model.local_head(local_cached)[offset:]) > 0.65)),
        ),
        "Level-2 relation GCN": (
            "model_only",
            lambda: model.conv2(torch.relu(model.conv1(local_cached, edge_index)), edge_index)[offset:],
        ),
        "Fusion": (
            "model_only",
            lambda: model.gate(torch.cat((local_cached, relation_cached), dim=1))[offset:],
        ),
        "total model full path": ("model_only", lambda: model(x, edge_index)[offset:]),
        "total end-to-end batch inference": (
            "end_to_end",
            run_end_to_end,
        ),
    }
    metadata = {
        "chain": chain,
        "seed": seed,
        "warmup_iterations": warmup,
        "batch_size": len(test_ids),
        "number_of_graphs": len(test_ids),
        "nodes_processed": int(x.shape[0]),
        "edges_processed": int(edge_index.shape[1]),
        "gpu_model": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "cuda_version": torch.version.cuda or "none",
        "pytorch_version": torch.__version__,
    }
    rows = []
    model.eval()
    with torch.no_grad():
        for stage, (scope, fn) in functions.items():
            rows.append({**metadata, "stage": stage, "scope": scope, "measurement_status": "measured", **measure(fn, device, warmup, repetitions)})
    for stage in (
        "event/state update",
        "local subgraph extraction",
        "node/edge feature preparation from raw event",
        "trace serialization",
        "cache/queue/checkpoint overhead",
    ):
        rows.append({**metadata, "stage": stage, "scope": "streaming_system", "measurement_status": "not_in_batch_evidence_path", "repetitions": 0, "mean_ms": np.nan, "median_ms": np.nan, "p95_ms": np.nan, "p99_ms": np.nan, "std_ms": np.nan})
    result = pd.DataFrame(rows)
    atomic_csv(output_dir / "end_to_end_stage_profile.csv", result)
    atomic_json(
        output_dir / "compute_scope_manifest.json",
        {
            **metadata,
            "architecture_under_test": "ContractDLG used by SCI-v2/v3 evidence runner",
            "level1_type": "11-D contract-summary MLP, not production Level1FraudGNN",
            "mc_scope": "Level-1 contract-summary encoder plus Level-1 fraud head",
            "old_component_benchmark_scope_warning": "legacy component_cost_benchmark labels local MLP as Level-1 and is not raw-event streaming E2E",
            "platform": platform.platform(),
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/mnt/d/_Work/_data/GoG_sci_v2")
    parser.add_argument("--output-dir", default="results/sci_v3_final/profiling")
    parser.add_argument("--chain", default="pooled")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    frame = run(Path(args.dataset_root), Path(args.output_dir), args.chain, args.seed, args.epochs, args.warmup, args.repetitions)
    print(json.dumps({"records": len(frame), "measured": int((frame.measurement_status == "measured").sum())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
