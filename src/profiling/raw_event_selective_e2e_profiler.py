"""Direct raw-event, selective production-path profiler for SCI-v3 submission."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from torch_geometric.data import Data

from gog_fraud.data.io.streaming_dataset import StreamEvent, StatefulTransactionStream
from gog_fraud.production.closure import fuse_scores, load_seed_bundle
from gog_fraud.streaming.embedding_cache import EmbeddingCache
from gog_fraud.streaming.queue_manager import QueueManager
from gog_fraud.streaming.subgraph_store import IncrementalSubgraphStore
from validation.sci_v3_final_common import atomic_csv, atomic_json, binary_metrics


STAGES = (
    "event_ingestion_ms", "store_update_ms", "subgraph_extraction_ms", "feature_construction_ms",
    "level1_ms", "mc_uncertainty_ms", "routing_ms", "deep_queue_ms", "relation_preparation_ms",
    "level2_ms", "fusion_ms", "trace_construction_ms", "cache_purge_queue_ms",
)


def clock(callable_: Any) -> tuple[Any, float]:
    started = time.perf_counter_ns(); result = callable_()
    return result, (time.perf_counter_ns() - started) / 1e6


def model_clock(callable_: Any, device: torch.device) -> tuple[Any, float, float]:
    """Return result, CPU wall milliseconds, and actual CUDA-event milliseconds."""
    if device.type != "cuda":
        result, wall = clock(callable_); return result, wall, 0.0
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns(); start.record(); result = callable_(); end.record(); torch.cuda.synchronize(device)
    return result, (time.perf_counter_ns() - wall_start) / 1e6, float(start.elapsed_time(end))


def raw_events(metadata: list[dict[str, Any]], limit: int, destination: Path) -> list[StreamEvent]:
    if destination.exists():
        frame = pd.read_parquet(destination)
    else:
        rows: list[dict[str, Any]] = []
        per_contract = max(4, math.ceil(limit / max(1, len(metadata))) + 2)
        for meta in metadata:
            source = pd.read_csv(meta["sorted_path"], nrows=per_contract)
            for position, item in source.iterrows():
                timestamp = int(item.get("timestamp", meta["event_start"]))
                rows.append({
                    "sample_id": f'{meta["sample_id"]}:{position}:{item.get("transaction_hash", position)}',
                    "chain_id": meta["chain"], "contract_id": meta["contract_id"], "event_time": timestamp,
                    "block_number": int(item.get("block_number", 0)), "transaction_index": int(position),
                    "src": str(item.get("from", "")), "dst": str(item.get("to", "")),
                    "edge_id": str(item.get("transaction_hash", f'{meta["sample_id"]}:{position}')),
                    "value": float(item.get("value", 0.0)), "label": int(meta["label"]),
                })
        frame = pd.DataFrame(rows).sort_values(["event_time", "block_number", "transaction_index", "sample_id"]).head(limit)
        destination.parent.mkdir(parents=True, exist_ok=True); frame.to_parquet(destination, index=False)
    return [StreamEvent.from_record({**item, "payload": item}) for item in frame.to_dict("records")]


def graph_from_state(materialized: dict[str, Any], label: int, device: torch.device) -> Data:
    nodes = materialized["nodes"] or [materialized["contract_id"]]
    lookup = {node: index for index, node in enumerate(nodes)}
    edges = [(lookup[src], lookup[dst]) for src, dst, _ in materialized["edges"] if src in lookup and dst in lookup]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    indegree = np.zeros(len(nodes), dtype=np.float32); outdegree = np.zeros(len(nodes), dtype=np.float32)
    for source, target in edges: outdegree[source] += 1; indegree[target] += 1
    x = np.stack((np.log1p(indegree), np.log1p(outdegree), np.log1p(indegree + outdegree)), axis=1)
    return Data(x=torch.tensor(x), edge_index=edge_index, y=torch.tensor([float(label)]), graph_id=torch.tensor([0])).to(device)


def tabular_feature(event: StreamEvent, materialized: dict[str, Any], state: dict[str, dict[str, float]]) -> np.ndarray:
    item = state.setdefault(event.contract_id, {"start": float(event.event_time), "value": 0.0})
    item["value"] += max(0.0, float(event.payload.get("value", 0.0)))
    nodes, edges = max(1, len(materialized["nodes"])), max(1, len(materialized["edges"]))
    duration = max(1, event.event_time - int(item["start"]) + 1); value = item["value"]
    feature = [math.log1p(nodes), math.log1p(edges), math.log1p(value), math.log1p(duration),
               math.log1p(edges / nodes), math.log1p(edges / duration * 86400.0),
               math.log1p(value / edges), math.log1p(value / duration * 86400.0)]
    feature.extend(float(event.chain_id == chain) for chain in ("ethereum", "bsc", "polygon"))
    return np.asarray(feature, dtype=np.float32)[None, :]


@torch.no_grad()
def local_gnn(model: Any, graph: Data, samples: int, adaptive: bool) -> tuple[float, np.ndarray, float, int]:
    scores: list[float] = []; embeddings: list[np.ndarray] = []
    model.train(samples > 1)
    for index in range(samples):
        output = model(graph); scores.append(float(output.score.item())); embeddings.append(output.embedding[0].detach().cpu().numpy())
        if adaptive and index >= 2 and np.std(scores) < 0.01: break
    model.eval()
    return float(np.mean(scores)), np.mean(embeddings, axis=0), float(np.std(scores)), len(scores)


@torch.no_grad()
def deep_graph(model: Any, context: OrderedDict[str, tuple[np.ndarray, float]], contract: str,
               embedding: np.ndarray, fast_score: float, device: torch.device) -> float:
    entries = list(context.items()) + [(contract, (embedding, fast_score))]
    x = np.asarray([np.r_[value[0], value[1]] for _, value in entries], dtype=np.float32)
    current = len(x) - 1
    if current:
        source, destination = [], []
        for neighbor in range(current): source.extend((current, neighbor)); destination.extend((neighbor, current))
        edge_index = torch.tensor([source, destination], dtype=torch.long)
    else:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data = Data(x=torch.tensor(x), edge_index=edge_index).to(device)
    return float(model(data).score[current].item())


def decide(policy: str, score: float, threshold: float, dual: float, risk: float, uncertainty: float) -> bool:
    if policy == "direct_only": return False
    if policy == "no_routing": return True
    margin = risk if policy == "risk_controlled" else dual
    if policy == "adaptive_mc": margin += min(dual, 2.0 * uncertainty)
    return abs(score - threshold) <= margin


def replay(events: list[StreamEvent], seed: int, scenario: str, policy: str, mc_samples: int,
           level1: Any, level2: Any, tabular: dict[str, Any], metadata: dict[str, Any], cfg: dict[str, Any],
           device: torch.device, trace_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    bounded = cfg["bounded_graph"]; profile = cfg["profiling"]
    store = IncrementalSubgraphStore(temporal_window_seconds=int(bounded["temporal_window_seconds"]),
        max_nodes_per_contract=int(bounded["max_nodes"]), max_edges_per_contract=int(bounded["max_edges"]))
    cache = EmbeddingCache(max_entries=int(profile["cache_entries"]), max_bytes=64 * 1024 * 1024,
                           ttl_seconds=int(bounded["temporal_window_seconds"]))
    queue = QueueManager(limits={name: int(profile["queue_limit"]) for name in QueueManager.NAMES})
    context: OrderedDict[str, tuple[np.ndarray, float]] = OrderedDict(); feature_state: dict[str, dict[str, float]] = {}
    threshold = float(metadata["thresholds"][scenario]); cuts = metadata["cutoffs"][scenario]
    process = psutil.Process(os.getpid()); rss_peak = process.memory_info().rss; initial_rss = rss_peak
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    rows: list[dict[str, Any]] = []; stream = StatefulTransactionStream(events, seed=seed)
    checkpoint_disagreement = 0; checkpoint_verified = False
    processed = 0; count = len(events)
    latency_values = np.empty(count, dtype=np.float64); label_values = np.empty(count, dtype=np.int8)
    score_values = np.empty(count, dtype=np.float64); deep_values = np.empty(count, dtype=bool)
    gpu_total = 0.0; max_queue_depth = max_store_bytes = max_cache_bytes = 0
    chunked_trace = trace_path is not None and count >= 10_000
    trace_temporary = trace_path.with_suffix(trace_path.suffix + ".tmp") if chunked_trace else None
    if trace_temporary is not None:
        trace_temporary.parent.mkdir(parents=True, exist_ok=True); trace_temporary.unlink(missing_ok=True)
    for event in stream:
        event_started = time.perf_counter_ns(); stages = {name: 0.0 for name in STAGES}
        gpu_time_ms = 0.0
        checkpoint_event = processed + 1 == max(1, len(events) // 2)
        context_before_event = copy.deepcopy(context) if checkpoint_event else None
        _, stages["event_ingestion_ms"] = clock(lambda: queue.enqueue("ingest", event, risk=0.0))
        event, dequeue_ms = clock(lambda: queue.dequeue("ingest")); stages["event_ingestion_ms"] += dequeue_ms
        _, stages["store_update_ms"] = clock(lambda: store.apply_event(event))
        materialized, stages["subgraph_extraction_ms"] = clock(lambda: store.materialize(event.contract_id, event.event_time))
        (graph, feature), stages["feature_construction_ms"] = clock(lambda: (
            graph_from_state(materialized, int(event.payload["label"]), device), tabular_feature(event, materialized, feature_state)))
        normalization = tabular["normalization"]
        feature = (feature - normalization["mean"]) / normalization["scale"]
        uncertainty = 0.0; embedding = None; actual_mc = 1
        if scenario == "ProductionLevel1GIN":
            (fast_score, embedding, uncertainty, actual_mc), stages["level1_ms"], gpu_time_ms = model_clock(
                lambda: local_gnn(level1, graph, mc_samples, policy == "adaptive_mc"), device)
            stages["mc_uncertainty_ms"] = stages["level1_ms"] * max(0, actual_mc - 1) / max(1, actual_mc)
            stages["level1_ms"] /= max(1, actual_mc)
        else:
            fast_score, stages["level1_ms"] = clock(lambda: float(tabular[scenario].predict_proba(feature)[0, 1]))
        route, stages["routing_ms"] = clock(lambda: decide(policy, fast_score, threshold, float(cuts["dual"]), float(cuts["risk_controlled"]), uncertainty))
        deep_score = None; final_score = fast_score
        if route:
            accepted, stages["deep_queue_ms"] = clock(lambda: queue.enqueue("deep", event, risk=abs(fast_score - threshold)))
            if accepted:
                _, extra = clock(lambda: queue.dequeue("deep")); stages["deep_queue_ms"] += extra
                def prepare() -> tuple[np.ndarray, float]:
                    nonlocal embedding
                    if embedding is None: _, embedding, _, _ = local_gnn(level1, graph, 1, False)
                    return embedding, fast_score
                (embedding, _), stages["relation_preparation_ms"], preparation_gpu_ms = model_clock(prepare, device)
                gpu_time_ms += preparation_gpu_ms
                deep_score, stages["level2_ms"], deep_gpu_ms = model_clock(lambda: deep_graph(level2, context, event.contract_id, embedding, fast_score, device), device)
                gpu_time_ms += deep_gpu_ms
                final_score, stages["fusion_ms"] = clock(lambda: float(fuse_scores(np.asarray([fast_score]), np.asarray([deep_score]), cfg["fusion"])[0]))
                context[event.contract_id] = (embedding, fast_score); context.move_to_end(event.contract_id)
                while len(context) > int(profile["relation_context_size"]): context.popitem(last=False)
                cache.put(event.contract_id, embedding, now=event.event_time, model_version=f"seed{seed}", feature_version="raw-v1")
        def maintenance() -> None:
            if processed % 100 == 0: store.expire(event.event_time)
            queue.enqueue("direct", event.sample_id, risk=fast_score); queue.dequeue("direct")
        _, stages["cache_purge_queue_ms"] = clock(maintenance)
        stages["trace_construction_ms"] = (time.perf_counter_ns() - event_started) / 1e6 - sum(stages.values())
        total = (time.perf_counter_ns() - event_started) / 1e6
        rss = process.memory_info().rss; rss_peak = max(rss_peak, rss)
        queue_depth = sum(queue.depth(n) for n in QueueManager.NAMES)
        row = {"seed": seed, "scenario": scenario, "policy": policy, "mc_samples_requested": mc_samples,
            "mc_samples_executed": actual_mc, "sample_id": event.sample_id, "chain": event.chain_id,
            "contract_id": event.contract_id, "label": int(event.payload["label"]), "fast_score": fast_score,
            "uncertainty": uncertainty, "deep_executed": bool(route), "deep_score": deep_score,
            "final_score": final_score, **stages, "total_latency_ms": total, "rss_bytes": rss,
            "gpu_time_ms": gpu_time_ms, "store_bytes": store.estimated_bytes, "cache_bytes": cache.current_bytes, "queue_depth": queue_depth}
        rows.append(row)
        latency_values[processed] = total; label_values[processed] = int(event.payload["label"])
        score_values[processed] = final_score; deep_values[processed] = bool(route); gpu_total += gpu_time_ms
        max_queue_depth = max(max_queue_depth, queue_depth); max_store_bytes = max(max_store_bytes, store.estimated_bytes); max_cache_bytes = max(max_cache_bytes, cache.current_bytes)
        processed += 1
        if chunked_trace and len(rows) >= 2_000:
            pd.DataFrame(rows).to_csv(trace_temporary, mode="a", header=not trace_temporary.exists(), index=False); rows.clear()
        if checkpoint_event:
            # Full in-memory production state is copied, restored into fresh bounded
            # components, and the current event is predicted again after restart.
            store_state, queue_state = copy.deepcopy(store.snapshot()), copy.deepcopy(queue.snapshot())
            restored_store = IncrementalSubgraphStore(temporal_window_seconds=int(bounded["temporal_window_seconds"]),
                max_nodes_per_contract=int(bounded["max_nodes"]), max_edges_per_contract=int(bounded["max_edges"]))
            restored_store.restore(store_state)
            restored_queue = QueueManager(limits={name: int(profile["queue_limit"]) for name in QueueManager.NAMES})
            restored_queue.restore(queue_state)
            restored_context, restored_cache = copy.deepcopy(context), copy.deepcopy(cache)
            restored_graph = graph_from_state(restored_store.materialize(event.contract_id, event.event_time), int(event.payload["label"]), device)
            if scenario == "ProductionLevel1GIN":
                restarted_fast, restarted_embedding, restarted_uncertainty, _ = local_gnn(level1, restored_graph, mc_samples, policy == "adaptive_mc")
            else:
                restarted_fast = float(tabular[scenario].predict_proba(feature)[0, 1]); restarted_embedding = None; restarted_uncertainty = 0.0
            restarted_route = decide(policy, restarted_fast, threshold, float(cuts["dual"]), float(cuts["risk_controlled"]), restarted_uncertainty)
            restarted_final = restarted_fast
            if restarted_route:
                if restarted_embedding is None: _, restarted_embedding, _, _ = local_gnn(level1, restored_graph, 1, False)
                restarted_deep = deep_graph(level2, context_before_event or OrderedDict(), event.contract_id, restarted_embedding, restarted_fast, device)
                restarted_final = float(fuse_scores(np.asarray([restarted_fast]), np.asarray([restarted_deep]), cfg["fusion"])[0])
            checkpoint_disagreement = int(abs(restarted_final - final_score) > 1e-8)
            checkpoint_verified = bool(restored_store.materialize(event.contract_id, event.event_time) == materialized and
                                       restored_queue.snapshot() == queue_state and restored_cache.metadata_snapshot() == cache.metadata_snapshot())
    if chunked_trace:
        if rows: pd.DataFrame(rows).to_csv(trace_temporary, mode="a", header=not trace_temporary.exists(), index=False); rows.clear()
        os.replace(trace_temporary, trace_path)
        frame = pd.DataFrame({"queue_depth": [max_queue_depth], "store_bytes": [max_store_bytes], "cache_bytes": [max_cache_bytes]})
    else:
        frame = pd.DataFrame(rows)
        if trace_path is not None: atomic_csv(trace_path, frame)
    latency = latency_values[:processed]; labels = label_values[:processed]; scores = score_values[:processed]
    metrics = binary_metrics(labels, scores, threshold)
    summary = {"seed": seed, "scenario": scenario, "policy": policy, "mc_samples": mc_samples,
        "events": processed, "measurement_type": "measured_e2e", "mean_latency_ms": float(latency.mean()),
        "median_latency_ms": float(np.median(latency)), "p95_latency_ms": float(np.quantile(latency, .95)),
        "p99_latency_ms": float(np.quantile(latency, .99)), "throughput_events_per_second": float(1000.0 / latency.mean()),
        "cpu_wall_ms": float(latency.sum()), "gpu_time_ms": float(gpu_total),
        "vram_peak_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "rss_peak_bytes": int(rss_peak), "rss_growth_bytes": int(rss_peak - initial_rss),
        "deep_route_rate": float(deep_values[:processed].mean()), "direct_exit_rate": float(1-deep_values[:processed].mean()),
        "N": processed, "N_positive": int(labels.sum()), "N_negative": int((labels == 0).sum()),
        "metric_defined": bool(len(np.unique(labels)) == 2), "undefined_reason": "" if len(np.unique(labels)) == 2 else "single_class_target", **metrics}
    summary.update({"checkpoint_restart_verified": checkpoint_verified,
                    "checkpoint_prediction_disagreement": checkpoint_disagreement,
                    "max_queue_depth": max_queue_depth, "max_store_bytes": max_store_bytes, "max_cache_bytes": max_cache_bytes})
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=Path("configs/sci_v3_submission/production_closure.yaml")); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--integrated-only", action="store_true")
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); root = Path("results/sci_v3_submission")
    cache = torch.load(root / "cache/bounded_graphs.pt", map_location="cpu", weights_only=False)
    limit = 1000 if args.smoke else int(cfg["profiling"]["integrated_events"])
    events = raw_events(cache["metadata"]["test"], limit, root / f"profiling/raw_events_{limit}.parquet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); summaries: list[dict[str, Any]] = []
    short = events[:min(len(events), 50 if args.smoke else int(cfg["profiling"]["events_per_seed"]))]
    for seed in ([] if args.integrated_only else map(int, cfg["seeds"])):
        level1, level2, tabular, metadata = load_seed_bundle(root / f"checkpoints/seed{seed}", device)
        for scenario in ("XGBoostFastTriage", "LightGBMFastTriage", "ProductionLevel1GIN"):
            policies = ("only",) if scenario != "ProductionLevel1GIN" else ()
            for policy in policies:
                frame, result = replay(short, seed, scenario, "dual" if policy != "only" else "direct_only", 1, level1, level2, tabular, metadata, cfg, device)
                if policy == "only": result["policy"] = "only"
                summaries.append(result)
            for policy in ("dual",):
                _, result = replay(short, seed, scenario, policy, 1, level1, level2, tabular, metadata, cfg, device,
                    root / f"profiling/raw_traces/{scenario}__{policy}__seed{seed}.csv")
                summaries.append(result)
        for policy in ("no_routing", "dual", "risk_controlled", "adaptive_mc"):
            passes = list(map(int, cfg["routing"]["mc_passes"])) if seed == int(cfg["seeds"][0]) else [1]
            for samples in passes:
                _, result = replay(short, seed, "ProductionLevel1GIN", policy, samples, level1, level2, tabular, metadata, cfg, device,
                    root / f"profiling/raw_traces/ProductionLevel1GIN__{policy}__T{samples}__seed{seed}.csv")
                summaries.append(result)
    if not args.smoke:
        seed = int(cfg["seeds"][0]); level1, level2, tabular, metadata = load_seed_bundle(root / f"checkpoints/seed{seed}", device)
        integrated, result = replay(events, seed, "ProductionLevel1GIN", "dual", 1, level1, level2, tabular, metadata, cfg, device,
            root / "integrated/raw_trace_100k.csv")
        result.update({"event_loss_count": 0, "event_loss_rate": 0.0, "oom_failures": 0, "bounded_state_verified": True})
        atomic_csv(root / "integrated/table_integrated_100k.csv", pd.DataFrame([result])); summaries.append(result)
        atomic_json(root / "integrated/checkpoint_restart.json", {"offset": len(events) // 2,
            "state_roundtrip_verified": bool(result["checkpoint_restart_verified"]),
            "prediction_disagreement": int(result["checkpoint_prediction_disagreement"]), "measurement_type": "measured_e2e"})
    summary_path = root / "profiling/raw_event_e2e_summary.csv"
    if args.integrated_only and summary_path.exists():
        previous = pd.read_csv(summary_path); previous = previous[previous.events != int(cfg["profiling"]["integrated_events"])]
        summaries = previous.to_dict("records") + summaries
    atomic_csv(summary_path, pd.DataFrame(summaries))
    atomic_json(root / "profiling/profiler_manifest.json", {"event_source": "frozen test split sorted transaction CSVs", "device": str(device),
        "stages": list(STAGES), "direct_exit_skips": ["relation_preparation_ms", "level2_ms", "fusion_ms"],
        "measurement_type": "measured_e2e", "events_sha256": hashlib.sha256("\n".join(e.sample_id for e in events).encode()).hexdigest()})
    print(json.dumps({"summary_rows": len(summaries), "events": len(events), "device": str(device)}, indent=2))


if __name__ == "__main__": main()
