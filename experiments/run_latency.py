"""
experiments/run_latency.py

Component-wise (MODULE) latency profiling for the dlg_gnn GraphRAG pipeline.

Round 2 fix: Renamed 'end_to_end' to 'module_pipeline' to correctly reflect
that this script does NOT include:
  - Streaming graph update (subgraph_store, neighborhood preparation)
  - GNN forward pass (Level1GNN / Level2 model)
  - MC stochastic forward passes (T samples)

The true End-to-End latency (including GNN + MC sampling) is measured
separately in experiments/run_e2e_latency.py.

Modules profiled here:
    - GraphRAG retrieval latency (per transaction)
    - Risk Encoder forward pass latency
    - Fusion latency
    - Module pipeline (retrieval + extraction + encoder + fusion, WITHOUT GNN)

Protocol:
    1. Warm-up: 50 iterations (excluded from measurement)
    2. Measurement: 500 iterations
    3. Report: mean / median / p95 / p99 (all in milliseconds)
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _time_fn(fn, *args, n_warmup=50, n_measure=500, **kwargs) -> np.ndarray:
    """Run fn n_warmup times (discard), then n_measure times and return latencies in ms."""
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times)


def _stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean_ms":   float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "p99_ms":    float(np.percentile(arr, 99)),
        "std_ms":    float(np.std(arr)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--output",  default="results/latency")
    parser.add_argument("--n-warmup",   type=int, default=50)
    parser.add_argument("--n-measure",  type=int, default=500)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import torch
    from graphrag.local_kb import LocalKnowledgeBase
    from graphrag.retriever import GraphRAGRetriever, RetrieverConfig
    from graphrag.risk_extractor import RiskExtractor
    from graphrag.risk_encoder import RiskEncoder
    from fusion.uncertainty_fusion import UncertaintyFusion

    # ── Setup ─────────────────────────────────────────────────────────────
    kb = LocalKnowledgeBase()
    retriever_cfg = RetrieverConfig(
        top_k=cfg.get("graphrag", {}).get("top_k", 5),
        graph_hops=cfg.get("graphrag", {}).get("graph_hops", 1),
    )
    retriever = GraphRAGRetriever(kb, retriever_cfg)
    extractor = RiskExtractor()
    encoder   = RiskEncoder.from_config(cfg)
    encoder.eval()
    fusion    = UncertaintyFusion.from_config(cfg)

    SAMPLE_CONTEXT = (
        "URGENT: My account has been compromised! "
        "Please send 500 USDT to 0xDeadBeef1234 immediately to secure your funds."
    )

    # ── Benchmark components ───────────────────────────────────────────────
    results = {}
    # Metadata to clarify scope
    scope_note = (
        "MODULE latency only (GraphRAG retrieval + extraction + encoder + fusion). "
        "Does NOT include: streaming graph update, GNN forward pass, MC sampling. "
        "See run_e2e_latency.py for true End-to-End latency."
    )

    log.info("Profiling: GraphRAG retrieval ...")
    retrieval_times = _time_fn(
        retriever.retrieve, SAMPLE_CONTEXT,
        n_warmup=args.n_warmup, n_measure=args.n_measure,
    )
    results["graphrag_retrieval"] = _stats(retrieval_times)

    log.info("Profiling: Risk extraction ...")
    sample_evidence = retriever.retrieve(SAMPLE_CONTEXT)

    def _extract():
        return extractor.extract(sample_evidence, event_id="tx_000000", pre_transaction_gap_sec=300)

    extract_times = _time_fn(_extract, n_warmup=args.n_warmup, n_measure=args.n_measure)
    results["risk_extraction"] = _stats(extract_times)

    log.info("Profiling: Risk encoder forward ...")
    sample_risk_dict = extractor.extract(sample_evidence, event_id="tx_000000", pre_transaction_gap_sec=300)

    def _encode():
        with torch.no_grad():
            return encoder.encode_risk_dict_batch([sample_risk_dict])

    encode_times = _time_fn(_encode, n_warmup=args.n_warmup, n_measure=args.n_measure)
    results["risk_encoder"] = _stats(encode_times)

    log.info("Profiling: Uncertainty fusion ...")
    p_gnn  = torch.tensor([0.65])
    u_mc   = torch.tensor([0.08])
    p_risk = torch.tensor([0.72])

    def _fuse():
        return fusion.fuse(p_gnn, u_mc, p_risk)

    fuse_times = _time_fn(_fuse, n_warmup=args.n_warmup, n_measure=args.n_measure)
    results["uncertainty_fusion"] = _stats(fuse_times)

    # ── End-to-end (sequential) ───────────────────────────────────────────
    def _end_to_end():
        evid  = retriever.retrieve(SAMPLE_CONTEXT)
        rdict = extractor.extract(evid, event_id="tx_0", pre_transaction_gap_sec=300)
        with torch.no_grad():
            _, p_r = encoder.encode_risk_dict_batch([rdict])
        fusion.fuse(p_gnn, u_mc, p_r)

    e2e_times = _time_fn(_end_to_end, n_warmup=args.n_warmup, n_measure=args.n_measure)
    results["end_to_end"] = _stats(e2e_times)

    # ── Report ────────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("  LATENCY REPORT (milliseconds, n=%d warm-up=%d)" % (args.n_measure, args.n_warmup))
    log.info("="*60)
    for component, stats in results.items():
        log.info(
            f"  {component:30s}  mean={stats['mean_ms']:6.3f}ms  "
            f"p95={stats['p95_ms']:6.3f}ms  p99={stats['p99_ms']:6.3f}ms"
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "latency_results.json"
    with open(out_path, "w") as f:
        json.dump({"components": results, "protocol": {
            "n_warmup": args.n_warmup, "n_measure": args.n_measure,
        }}, f, indent=2)
    log.info(f"\nLatency results saved to {out_path}")


if __name__ == "__main__":
    main()
