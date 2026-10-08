#!/usr/bin/env python3
"""Fixed one-block A05 AnomalyDAE projection before large-graph runs."""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from gog_fraud.evaluation.reproducibility import seed_everything
from gog_fraud.pipelines.run_sci_round4c import _instantiate, _models

OUT = ROOT / "evaluation/benchmark/v2/diagnostics/anomalydae_chunked"
CONFIG = ROOT / "configs/benchmark/sci_round5_final.yaml"
BLOCK = 256
LARGE_SAMPLE_BLOCKS = 16  # fixed refinement for N>100000 before full production runs
GUARD_SECONDS = 24 * 3600
DATASETS = ("Ethereum", "DGraphFin", "Yelp-Syn", "Reddit-Syn")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("dataset", choices=DATASETS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"preflight_{args.dataset}.json"
    report = {"dataset": args.dataset, "seed": 42, "block_size": BLOCK,
              "projection_guard_seconds": GUARD_SECONDS,
              "measurement_plan": "one encoder forward; one block for N<=100000, first 16 consecutive "
                                  "256-row blocks for N>100000 (discard block 1 as warmup); "
                                  "50*ceil(N/256) blocks is a lower-bound projection"}
    try:
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        if "RTX 3090" not in report["gpu_name"]:
            raise RuntimeError(f"A05 large-graph preflight requires RTX 3090, got {report['gpu_name']}")
        graph = torch.load(ROOT / f"outputs/benchmark/a04_constructed_graphs/{args.dataset}.pt",
                           map_location="cpu", weights_only=False)
        data = Data(x=graph["x"], edge_index=graph["edge_index"], y=graph["y"],
                    num_nodes=graph["num_nodes"])
        config = yaml.safe_load(CONFIG.read_text())
        seed_everything(42, deterministic=True)
        detector = _instantiate(config, "AnomalyDAE", _models(config)["AnomalyDAE"], 0)
        detector.process_graph(data)
        detector.num_nodes, detector.in_dim = data.x.shape
        detector.batch_size = detector.num_nodes
        detector.model = detector.init_model(**detector.kwargs)
        detector.model.train()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
        t0 = time.perf_counter()
        components = detector._forward_components(data)
        torch.cuda.synchronize(0)
        encoder_seconds = time.perf_counter() - t0
        sample_count = LARGE_SAMPLE_BLOCKS if data.num_nodes > 100000 else 1
        block_times=[]
        finite=True
        for sample in range(sample_count):
            rows = torch.arange(sample*BLOCK, min((sample+1)*BLOCK, data.num_nodes), device=detector.device)
            t0 = time.perf_counter()
            score = detector._score_rows(components, rows)
            loss = score.sum() / data.num_nodes
            loss.backward(retain_graph=sample < sample_count-1)
            torch.cuda.synchronize(0)
            block_times.append(time.perf_counter() - t0)
            finite = finite and bool(torch.isfinite(score).all())
        block_seconds = sum(block_times[1:]) / (sample_count-1) if sample_count>1 else block_times[0]
        blocks = math.ceil(data.num_nodes / BLOCK)
        lower_bound_seconds = 50 * (encoder_seconds + blocks * block_seconds)
        status = "UNSUPPORTED_OPERATIONAL_TIMEOUT" if lower_bound_seconds > GUARD_SECONDS else "PREFLIGHT_FEASIBLE"
        report.update({"status": status, "nodes": data.num_nodes, "edges": data.num_edges,
                       "encoder_forward_seconds": encoder_seconds,
                       "first_block_forward_backward_seconds": block_times[0],
                       "projected_block_seconds": block_seconds,
                       "sampled_blocks":sample_count,"sample_block_seconds":block_times,
                       "blocks_per_epoch": blocks, "projected_50_epoch_lower_bound_seconds": lower_bound_seconds,
                       "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 2**20,
                       "finite_score": finite})
    except BaseException as exc:
        report.update({"status": "UNSUPPORTED_RESOURCE_OOM" if isinstance(exc, torch.OutOfMemoryError) or
                       "out of memory" in str(exc).lower() else "PREFLIGHT_ERROR",
                       "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k,v in report.items() if k != "traceback"}), flush=True)


if __name__ == "__main__": main()
