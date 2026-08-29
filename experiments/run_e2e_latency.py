"""
experiments/run_e2e_latency.py

True End-to-End (E2E) latency profiling for the dlg_gnn pipeline.

Round 2 requirement (TASK 3.2-B): Measure the FULL inference path including:
  1. Streaming graph update / neighborhood preparation (simulated sub-graph lookup)
  2. GNN forward pass (Level1GNN model, or simulated if no checkpoint available)
  3. T Monte Carlo stochastic forward passes (MC dropout)
  4. GraphRAG retrieval
  5. Risk extraction
  6. Risk encoder
  7. Uncertainty fusion

T values: [1, 5, 10, 20, 30]

Outputs:
    results/e2e_latency_by_T.csv
    reports/hardware_profile.md

Usage:
    python experiments/run_e2e_latency.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

T_VALUES = [1, 5, 10, 20, 30]
N_WARMUP = 30
N_MEASURE = 200  # fewer reps as each measurement is more expensive


def _stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "std_ms": float(np.std(arr)),
        "throughput_eps": float(1000.0 / np.mean(arr)),  # events per second
    }


def _collect_hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else "N/A"
        if torch.cuda.is_available():
            info["gpu_model"] = torch.cuda.get_device_name(0)
        else:
            info["gpu_model"] = "N/A"
    except ImportError:
        info["torch_version"] = "N/A"

    info["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    info["platform"] = platform.platform()

    try:
        cpu_raw = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in cpu_raw.split("\n"):
            if "Model name" in line:
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except Exception:
        info["cpu_model"] = platform.processor()

    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / 1024**3, 1)
    except ImportError:
        info["ram_total_gb"] = "N/A"

    info["n_warmup_iters"] = N_WARMUP
    info["n_measure_iters"] = N_MEASURE

    return info


class SimulatedGNN:
    """
    Simulates a Level1GNN forward pass for latency profiling purposes.

    In the absence of a trained checkpoint, we simulate the computational
    cost of:
      - A 3-layer GraphSAGE with hidden_dim=128
      - A batch of N_NODES nodes per transaction subgraph
      - A linear MLP head

    This provides realistic latency estimates for CPU-bound inference.
    The 'simulated' flag is recorded in all output artifacts.
    """

    def __init__(self, hidden_dim: int = 128, n_layers: int = 3, n_nodes: int = 15):
        import torch
        import torch.nn as nn
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.head = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.is_simulated = True

    def forward_mc(self, T: int, rng: np.random.RandomState) -> tuple[float, float]:
        """Run T stochastic forward passes, return (mean_score, uncertainty)."""
        import torch
        import torch.nn.functional as F
        scores = []
        x = torch.from_numpy(
            rng.randn(self.n_nodes, self.hidden_dim).astype(np.float32)
        )
        for _ in range(T):
            h = x
            for layer in self.layers:
                h = self.relu(layer(h))
                # Simulate MC dropout: randomly zero 20% of activations
                mask = torch.from_numpy(
                    (rng.rand(*h.shape) > 0.2).astype(np.float32)
                )
                h = h * mask
            logit = self.head(h).mean(dim=0)
            scores.append(float(torch.sigmoid(logit).item()))
        arr = np.array(scores)
        return float(arr.mean()), float(arr.var())


def run_e2e_for_T(T: int, pipeline_components: dict) -> np.ndarray:
    """
    Measure N_MEASURE E2E event latencies for a given T (MC samples).
    Returns array of latencies in ms.
    """
    retriever = pipeline_components["retriever"]
    extractor = pipeline_components["extractor"]
    encoder = pipeline_components["encoder"]
    fusion = pipeline_components["fusion"]
    gnn = pipeline_components["gnn"]
    sample_context = pipeline_components["sample_context"]
    rng = np.random.RandomState(42 + T * 7)

    import torch

    # Warm-up
    for _ in range(N_WARMUP):
        # 1. Graph update (simulated: small dict lookup)
        _ = {"nodes": rng.randint(0, 100, 15), "edges": rng.randint(0, 100, 20)}
        # 2. GNN forward (T MC passes)
        gnn.forward_mc(T, rng)
        # 3. GraphRAG
        evid = retriever.retrieve(sample_context)
        rdict = extractor.extract(evid, event_id="tx_warmup", pre_transaction_gap_sec=300)
        with torch.no_grad():
            _, p_r = encoder.encode_risk_dict_batch([rdict])
        fusion.fuse(torch.tensor([0.5]), torch.tensor([0.1]), p_r)

    # Measurement
    times = []
    for i in range(N_MEASURE):
        t0 = time.perf_counter()

        # Step 1: Streaming graph update (subgraph store lookup)
        _graph_state = {"nodes": rng.randint(0, 100, 15), "edges": rng.randint(0, 100, 20)}

        # Step 2: GNN forward pass + T MC stochastic passes
        p_bar, u_t = gnn.forward_mc(T, rng)

        # Step 3: GraphRAG retrieval
        evid = retriever.retrieve(sample_context)

        # Step 4: Risk extraction
        rdict = extractor.extract(evid, event_id=f"tx_{i:06d}", pre_transaction_gap_sec=300)

        # Step 5: Risk encoder
        with torch.no_grad():
            _, p_r = encoder.encode_risk_dict_batch([rdict])

        # Step 6: Uncertainty fusion
        fusion.fuse(
            torch.tensor([float(p_bar)]),
            torch.tensor([float(u_t)]),
            p_r,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed_ms)

    return np.array(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-csv", default="results/e2e_latency_by_T.csv")
    parser.add_argument("--report", default="reports/hardware_profile.md")
    parser.add_argument(
        "--T-values",
        default=",".join(str(t) for t in T_VALUES),
        help="Comma-separated MC sample counts to sweep",
    )
    args = parser.parse_args()

    T_list = [int(t) for t in args.T_values.split(",")]

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

    kb = LocalKnowledgeBase()
    retriever_cfg = RetrieverConfig(
        top_k=cfg.get("graphrag", {}).get("top_k", 5),
        graph_hops=cfg.get("graphrag", {}).get("graph_hops", 1),
    )
    retriever = GraphRAGRetriever(kb, retriever_cfg)
    extractor = RiskExtractor()
    encoder = RiskEncoder.from_config(cfg)
    encoder.eval()
    fusion = UncertaintyFusion.from_config(cfg)
    gnn = SimulatedGNN(hidden_dim=128, n_layers=3, n_nodes=15)

    sample_context = (
        "URGENT: My account has been compromised! "
        "Please send 500 USDT to 0xDeadBeef1234 immediately to secure your funds."
    )

    pipeline_components = dict(
        retriever=retriever,
        extractor=extractor,
        encoder=encoder,
        fusion=fusion,
        gnn=gnn,
        sample_context=sample_context,
    )

    # Collect hardware info
    hw = _collect_hardware_info()
    hw["gnn_model"] = "SimulatedGNN (3-layer GraphSAGE proxy, CPU)"
    hw["gnn_n_nodes_per_event"] = gnn.n_nodes
    hw["graph_update_simulation"] = "dict lookup (no real subgraph_store in profile)"
    hw["gnn_checkpoint"] = "simulated — no trained checkpoint used"
    hw["n_seeds_measured"] = 1
    hw["T_values"] = T_list

    # Sweep T values
    rows = []
    for T in T_list:
        log.info(f"Profiling E2E latency for T={T} MC samples ...")
        latencies = run_e2e_for_T(T, pipeline_components)
        stats = _stats(latencies)
        row = {
            "T": T,
            "gnn_source": "simulated",
            "n_measure": N_MEASURE,
            "n_warmup": N_WARMUP,
            **{k: round(v, 4) for k, v in stats.items()},
        }
        rows.append(row)
        log.info(
            f"  T={T:2d}: mean={stats['mean_ms']:.3f}ms  "
            f"p95={stats['p95_ms']:.3f}ms  "
            f"throughput={stats['throughput_eps']:.0f} ev/s"
        )

    # Save CSV
    import pandas as pd
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    log.info(f"Saved: {out_csv}")

    # Save JSON
    json_path = out_csv.parent / "e2e_latency_by_T.json"
    with open(json_path, "w") as f:
        json.dump({"hardware": hw, "results": rows}, f, indent=2)
    log.info(f"Saved: {json_path}")

    # Generate hardware_profile.md
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Hardware Profile\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## System\n\n")
        for k, v in hw.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## E2E Latency by T (MC Samples)\n\n")
        f.write("| T | mean_ms | median_ms | p95_ms | p99_ms | throughput_eps |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['T']} | {r['mean_ms']:.3f} | {r['median_ms']:.3f} | "
                f"{r['p95_ms']:.3f} | {r['p99_ms']:.3f} | {r['throughput_eps']:.1f} |\n"
            )
        f.write("\n> **Note:** GNN component is simulated (no trained checkpoint). ")
        f.write("Graph update is a dict-lookup proxy. ")
        f.write("These numbers reflect the MC overhead scaling, not absolute real-world latency.\n")
    log.info(f"Saved: {report_path}")

    # Print summary table
    log.info("\n" + "=" * 70)
    log.info("  TRUE E2E LATENCY BY T (GNN simulation + full module pipeline)")
    log.info("  ⚠️  GNN component: SIMULATED (no trained checkpoint)")
    log.info("=" * 70)
    log.info(f"  {'T':>4}  {'mean':>8}  {'p95':>8}  {'p99':>8}  {'ev/s':>8}")
    for r in rows:
        log.info(
            f"  {r['T']:>4}  {r['mean_ms']:>7.3f}ms  "
            f"{r['p95_ms']:>7.3f}ms  {r['p99_ms']:>7.3f}ms  "
            f"{r['throughput_eps']:>7.1f}"
        )


if __name__ == "__main__":
    main()
