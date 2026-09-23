from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gog_fraud.data.io.streaming_dataset import StatefulTransactionStream
from gog_fraud.data.splits.artifact import scan_contract_records
from gog_fraud.selection.router import SelectiveRouter, TriageOutput
from gog_fraud.streaming.checkpoint import PipelineCheckpoint, save_checkpoint
from gog_fraud.streaming.engine import StatefulStreamingEngine
from profiling.streaming_profiler import StreamingProfiler


def _score(sample_id: str) -> float:
    # Stable non-label heuristic used only to exercise the path, never as a paper metric.
    return int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction-root", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chains", nargs="+", default=["ethereum", "bsc", "polygon"])
    parser.add_argument("--samples-per-chain", type=int, default=100)
    args = parser.parse_args()
    output = Path(args.output_root); output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for chain in args.chains:
        records, exclusions = scan_contract_records(args.transaction_root, args.labels, chain, max_files=args.samples_per_chain)
        stream_records = [
            {"sample_id": row["sample_id"], "chain_id": chain, "contract_id": row["sample_id"],
             "event_time": row["event_time"], "payload": {"transaction_count": row["transaction_count"]}}
            for row in records
        ]
        stream = StatefulTransactionStream(stream_records)
        router = SelectiveRouter(tau_b=0.2, tau_f=0.8, tau_u=0.04, threshold_version="smoke-only-v1")
        def triage(event):
            score = _score(event.sample_id)
            variance = min(0.1, 1.0 / math.sqrt(max(1, event.payload["transaction_count"])))
            return TriageOutput(score, variance, math.sqrt(variance), 0.0, None, 8)
        engine = StatefulStreamingEngine(stream=stream, router=router, triage_fn=triage, deep_fn=lambda event: _score(event.sample_id), model_version="non_model_smoke_heuristic")
        profiler = StreamingProfiler()
        profiler.record_memory(0)
        with profiler.timer("end_to_end", cold_start=True):
            traces = engine.run()
        profiler.record_memory(len(traces))
        trace_path = output / f"{chain}_sample_trace.jsonl"
        trace_path.write_text("".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in traces), encoding="utf-8")
        profiler_path = output / f"{chain}_profiler_trace.json"; profiler.write_trace(profiler_path)
        checkpoint = PipelineCheckpoint(
            stream=asdict(stream.checkpoint()), subgraph_store={}, relation_state=[], embedding_cache={}, queues={},
            model_version="non_model_smoke_heuristic", threshold_version="smoke-only-v1",
        )
        checkpoint_path = output / f"{chain}_checkpoint.json"
        checkpoint_hash = save_checkpoint(checkpoint, checkpoint_path)
        summaries.append({
            "chain": chain, "status": "PASS", "scientific_status": "SMOKE_ONLY_NOT_PAPER_ELIGIBLE",
            "samples_requested": args.samples_per_chain, "samples_processed": len(traces), "excluded": exclusions,
            "routing": engine.routing_summary(), "prediction_hash": engine.prediction_hash,
            "latency": profiler.latency_summary("end_to_end"), "peak_rss_mb": profiler.get_peak_memory_mb(),
            "memory_slope_mb_per_10k": profiler.memory_slope_mb_per_10k(),
            "checkpoint_hash": checkpoint_hash, "trace": str(trace_path), "profiler_trace": str(profiler_path),
            "paper_metrics_generated": False,
        })
    (output / "smoke_summary.json").write_text(json.dumps(summaries, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
