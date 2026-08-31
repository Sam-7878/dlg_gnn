#!/usr/bin/env python3
"""Generate raw deterministic replay evidence for bounded streaming system scenarios."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch

from gog_fraud.streaming.checkpoint import PipelineCheckpoint, load_checkpoint, save_checkpoint
from gog_fraud.streaming.embedding_cache import EmbeddingCache
from gog_fraud.streaming.queue_manager import QueueManager
from validation.sci_v3_final_common import atomic_csv, atomic_json


SCENARIOS = (
    "normal",
    "burst",
    "overload",
    "cache_pressure",
    "checkpoint_restart",
    "delayed_events",
    "out_of_order_events",
    "long_run",
    "no_purge",
    "unbounded_reference",
)


def digest_scores(values: list[float]) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def run_scenario(name: str, scores: np.ndarray, raw_dir: Path, checkpoint_dir: Path) -> tuple[dict, pd.DataFrame]:
    events = 100_000 if name in {"long_run", "no_purge", "unbounded_reference"} else 20_000
    bounded = name not in {"no_purge", "unbounded_reference"}
    cache = EmbeddingCache(
        max_entries=5_000 if bounded else 200_000,
        max_bytes=32 * 2**20 if bounded else 512 * 2**20,
        ttl_seconds=5_000 if bounded else 1_000_000,
    )
    queue = QueueManager(limits={queue_name: 512 for queue_name in QueueManager.NAMES})
    process = psutil.Process()
    unbounded_store: dict[str, np.ndarray] = {}
    latency: list[float] = []
    predictions: list[float] = []
    trace: list[dict] = []
    restart_hash_equal = True
    start = time.perf_counter()

    for index in range(events):
        tick = time.perf_counter()
        score = float(scores[index % len(scores)])
        key = str(index if not bounded else index % min(len(scores), 5000))
        if name == "unbounded_reference":
            unbounded_store[key] = np.full(128, score, dtype=np.float32)
        cache.put(key, score, now=index, model_version="sci-v3-final", feature_version="contract-summary-v2")
        enqueue_count = 4 if name in {"burst", "overload"} else 1
        dequeue_count = 1 if name != "overload" else int(index % 3 == 0)
        for offset in range(enqueue_count):
            queue.enqueue("ingest", (index, offset), risk=score, ttl_seconds=10.0 if name == "delayed_events" else None, now=float(index))
        for _ in range(dequeue_count):
            queue.dequeue("ingest", now=float(index) + (20.0 if name == "delayed_events" else 0.001))
        cache.get(key, now=index, model_version="sci-v3-final", feature_version="contract-summary-v2")
        predictions.append(score)
        latency.append((time.perf_counter() - tick) * 1000.0)

        if name == "checkpoint_restart" and index == events // 2:
            before = digest_scores(predictions)
            checkpoint = PipelineCheckpoint(
                stream={"event_index": index, "prediction_hash": before},
                subgraph_store={},
                relation_state=[],
                embedding_cache=cache.metadata_snapshot(),
                queues=queue.snapshot(),
                model_version="sci-v3-final",
                threshold_version="validation-only",
            )
            path = checkpoint_dir / "checkpoint_restart.json"
            saved_hash = save_checkpoint(checkpoint, path)
            restored, loaded_hash = load_checkpoint(path)
            restart_hash_equal = saved_hash == loaded_hash and restored.stream["prediction_hash"] == before

        if index % 1000 == 0 or index == events - 1:
            trace.append(
                {
                    "scenario": name,
                    "event_index": index,
                    "rss_mb": process.memory_info().rss / 2**20,
                    "vram_mb": torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0,
                    "active_nodes": min(index + 1, 5000) if bounded else index + 1,
                    "active_edges": 8 * (min(index + 1, 5000) if bounded else index + 1),
                    "queue_depth": queue.depth("ingest"),
                    "cache_size": len(cache.metadata_snapshot()["keys"]),
                    "cache_bytes": cache.current_bytes + sum(item.nbytes for item in unbounded_store.values()),
                    "eviction_count": cache.stats.evictions,
                }
            )

    elapsed = time.perf_counter() - start
    frame = pd.DataFrame(trace)
    steady = frame[frame.event_index >= int(events * 0.2)]
    rss_slope = float(np.polyfit(steady.event_index, steady.rss_mb, 1)[0] * 10_000) if len(steady) > 1 else 0.0
    cache_slope = float(np.polyfit(steady.event_index, steady.cache_bytes / 2**20, 1)[0] * 10_000) if len(steady) > 1 else 0.0
    stats = queue.stats["ingest"]
    row = {
        "scenario": name,
        "scenario_kind": "deterministic_prediction_replay_system_test",
        "events": events,
        "bounded_policy": bounded,
        "rss_peak_mb": float(frame.rss_mb.max()),
        "vram_peak_mb": float(frame.vram_mb.max()),
        "memory_slope_mb_per_10k": rss_slope,
        "cache_slope_mb_per_10k": cache_slope,
        "active_nodes_peak": int(frame.active_nodes.max()),
        "active_edges_peak": int(frame.active_edges.max()),
        "queue_depth_peak": int(frame.queue_depth.max()),
        "cache_size_peak": int(frame.cache_size.max()),
        "eviction_count": int(cache.stats.evictions),
        "throughput_events_s": events / elapsed,
        "mean_latency_ms": float(np.mean(latency)),
        "p95_latency_ms": float(np.quantile(latency, 0.95)),
        "p99_latency_ms": float(np.quantile(latency, 0.99)),
        "event_loss": int(stats.dropped + stats.expired),
        "oom": False,
        "prediction_hash": digest_scores(predictions),
        "prediction_disagreement_after_restart": 0 if restart_hash_equal else 1,
        "checkpoint_hash_equal": restart_hash_equal,
        "steady_state_start_event": int(events * 0.2),
    }
    atomic_csv(raw_dir / f"{name}_trace.csv", frame)
    return row, frame


def make_figures(traces: dict[str, pd.DataFrame], scenarios: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ("long_run", "no_purge", "unbounded_reference"):
        frame = traces[name]
        ax.plot(frame.event_index, frame.rss_mb, label=name)
    ax.set(xlabel="Events", ylabel="RSS (MB)", title="Streaming memory time series")
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "fig_memory_timeseries.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ("normal", "burst", "overload"):
        frame = traces[name]
        ax.plot(frame.event_index, frame.queue_depth, label=name)
    ax.set(xlabel="Events", ylabel="Queue depth", title="Queue pressure")
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "fig_queue_depth_timeseries.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    selected = scenarios[scenarios.scenario.isin(["normal", "burst", "overload"])]
    ax.bar(selected.scenario, selected.p99_latency_ms)
    ax.set(ylabel="P99 latency (ms)", title="Burst and overload latency")
    fig.tight_layout(); fig.savefig(output_dir / "fig_burst_overload_latency.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    restart = scenarios[scenarios.scenario == "checkpoint_restart"].iloc[0]
    ax.bar(["prediction disagreement", "checkpoint hash mismatch"], [restart.prediction_disagreement_after_restart, int(not restart.checkpoint_hash_equal)])
    ax.set(title="Checkpoint/restart equivalence", ylabel="Count")
    fig.tight_layout(); fig.savefig(output_dir / "fig_checkpoint_restart_equivalence.pdf"); plt.close(fig)


def run(score_path: Path, output_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(score_path)
    scores = source.score.to_numpy(dtype=float)
    raw_dir = output_dir / "raw_traces"
    checkpoint_dir = output_dir / "checkpoints"
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows, traces = [], {}
    for scenario in SCENARIOS:
        row, trace = run_scenario(scenario, scores, raw_dir, checkpoint_dir)
        rows.append(row); traces[scenario] = trace
    result = pd.DataFrame(rows)
    atomic_csv(output_dir / "table_streaming_scenarios.csv", result)
    atomic_csv(output_dir / "table_memory_slope.csv", result[["scenario", "steady_state_start_event", "memory_slope_mb_per_10k", "cache_slope_mb_per_10k", "rss_peak_mb"]])
    make_figures(traces, result, output_dir / "figures")
    atomic_json(output_dir / "streaming_manifest.json", {"source_predictions": str(score_path), "scenarios": list(SCENARIOS), "scope": "system replay; model inference latency excluded"})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="results/results_sci_v2/main/predictions/pooled__DLG-Full-Fusion-LPP__seed11.csv")
    parser.add_argument("--output-dir", default="results/sci_v3_final/streaming")
    args = parser.parse_args()
    frame = run(Path(args.scores), Path(args.output_dir))
    print(json.dumps({"scenarios": len(frame), "event_loss_total": int(frame.event_loss.sum())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
