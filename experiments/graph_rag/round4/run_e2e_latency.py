"""Measure context-free SCI main-track latency using a real checkpoint."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.round4.artifact_paths import CHECKPOINT_DIR, DATASET_DIR, RESULTS_DIR, ensure_dirs
from experiments.round4.data import load_packed
from experiments.round4.model import CausalLocalGIN


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(device: torch.device) -> CausalLocalGIN:
    payload = torch.load(CHECKPOINT_DIR / "seed7.pt", map_location="cpu", weights_only=False)
    config = payload["model_config"]
    model = CausalLocalGIN(config["input_dim"], config["hidden_dim"], config["dropout"])
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); ensure_dirs()
    _, manifest, datasets = load_packed(DATASET_DIR)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = load_model(device)
    graphs = datasets["test"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Warm-up is explicitly excluded.
    warm = Batch.from_data_list([graphs[0]]).to(device)
    for _ in range(20):
        model.forward_mc(warm, 10)
    sync(device)
    rows = []
    for passes in (1, 5, 10, 20, 30):
        component = {"historical_graph_retrieval_update_ms": [], "gnn_mc_forward_ms": [], "serialization_ms": [], "total_ms": []}
        for index in range(min(args.samples, len(graphs))):
            total_start = time.perf_counter()
            start = time.perf_counter()
            graph = graphs[index]
            batch = Batch.from_data_list([graph]).to(device)
            sync(device)
            component["historical_graph_retrieval_update_ms"].append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            if passes == 1:
                model.eval()
                with torch.no_grad():
                    probability = torch.sigmoid(model(batch)); variance = torch.zeros_like(probability)
            else:
                _, probability, variance, _ = model.forward_mc(batch, passes)
            sync(device)
            component["gnn_mc_forward_ms"].append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            json.dumps({"event_index": index, "probability": float(probability.item()),
                        "variance": float(variance.item()), "T": passes})
            component["serialization_ms"].append((time.perf_counter() - start) * 1000)
            component["total_ms"].append((time.perf_counter() - total_start) * 1000)
        total = np.asarray(component["total_ms"])
        row = {
            "track": "SCI Main Track", "seed": 7, "T": passes, "n_events": len(total),
            "gnn_source": "real_checkpoint", "split_type": manifest["split_type"],
            "context_branch_enabled": False, "graphrag_retrieval_ms": 0.0,
            "risk_encoder_ms": 0.0, "fusion_ms": 0.0,
            **{key: statistics.mean(values) for key, values in component.items() if key != "total_ms"},
            "mean_ms": float(total.mean()), "median_ms": float(np.median(total)),
            "p95_ms": float(np.quantile(total, 0.95)), "p99_ms": float(np.quantile(total, 0.99)),
            "events_per_second": float(1000 / total.mean()),
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        }
        rows.append(row)
        print(f"T={passes} mean_ms={row['mean_ms']:.3f} p95_ms={row['p95_ms']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "e2e_latency.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
