"""
Phase J: True Real End-to-End Latency Measurement

Measures true E2E latency including:
- Graph neighborhood retrieval (GoG-MicroRAG)
- Real trained GNN forward (T MC passes)
- GraphRAG retrieval
- Risk extraction + encoding
- Fusion

Usage:
    python experiments/round3/run_real_e2e_latency.py

Outputs:
    results/real_e2e_latency.csv
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data as PyGData

ROOT = Path(__file__).parent.parent.parent
if not (ROOT / "data").exists():
    ROOT = Path(os.environ.get("DLG_GNN_ROOT", "/mnt/d/_Work/goat_bank/dlg_gnn"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("real_e2e_latency")

DATA_DIR = ROOT / "data" / "benchmark" / "gog_microrag_stream_v1"
from experiments.round3.artifact_paths import CHECKPOINT_DIR as CKPT_DIR, ROUND3_RESULTS as RESULTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Inline model classes (same as train_gog_l1_v2.py)
# ─────────────────────────────────────────────────────────────────────────────

class GINLayerV2(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2), nn.BatchNorm1d(out_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(out_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim), nn.GELU(),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        agg = torch.zeros_like(x)
        agg.index_add_(0, edge_index[0], x[edge_index[1]])
        return self.net((1 + self.eps) * x + agg) + self.res(x)


class Level1GNNv2(nn.Module):
    def __init__(self, in_dim=21, hidden_dim=256, num_layers=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
        )
        self.layers = nn.ModuleList([GINLayerV2(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, data):
        h = self.input_proj(data.x)
        h_list = [h]
        for layer in self.layers:
            h = layer(h, data.edge_index)
            h_list.append(h)
        return self.head(torch.cat(h_list[-3:], dim=-1)).squeeze(-1)

    def forward_mc(self, data, T=10):
        self.train()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        probs_list = []
        with torch.no_grad():
            for _ in range(T):
                probs_list.append(torch.sigmoid(self.forward(data)))
        self.eval()
        probs = torch.stack(probs_list)
        mean_p = probs.mean(0)
        variance = probs.var(0, unbiased=False)
        eps = 1e-8
        entropy = -(mean_p * torch.log(mean_p + eps) + (1 - mean_p) * torch.log(1 - mean_p + eps))
        return mean_p, variance, entropy


class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(out_dim, out_dim), nn.ReLU(),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index):
        agg = torch.zeros_like(x)
        agg.index_add_(0, edge_index[0], x[edge_index[1]])
        return self.net((1 + self.eps) * x + agg)


class Level1GNNDirect(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.layers.append(GINLayer(dims[i], dims[i+1], dropout))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        h = data.x
        for layer in self.layers:
            h = layer(h, data.edge_index)
        return self.head(torch.cat([h, h], dim=-1)).squeeze(-1)

    def forward_mc(self, data, T=10):
        self.train()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        probs_list = []
        with torch.no_grad():
            for _ in range(T):
                probs_list.append(torch.sigmoid(self.forward(data)))
        self.eval()
        return torch.stack(probs_list).mean(0), torch.stack(probs_list).var(0), torch.zeros_like(data.x[:, 0])


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering (same as v2)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(graph):
    emb = graph["embeddings"].float()
    ei = graph["edge_index"]
    N = emb.shape[0]
    in_deg = torch.zeros(N)
    out_deg = torch.zeros(N)
    in_deg.index_add_(0, ei[1], torch.ones(ei.shape[1]))
    out_deg.index_add_(0, ei[0], torch.ones(ei.shape[1]))
    total_deg = in_deg + out_deg
    log_in = torch.log1p(in_deg).unsqueeze(1)
    log_out = torch.log1p(out_deg).unsqueeze(1)
    log_total = torch.log1p(total_deg).unsqueeze(1)
    norm = emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
    emb_norm = emb / norm
    emb_max = emb.max(dim=1)[0].unsqueeze(1)
    emb_sum = emb.sum(dim=1).unsqueeze(1)
    return torch.cat([emb, log_in, log_out, log_total, emb_max, emb_sum, emb_norm], dim=1)


def build_single_node_data(graph, node_id, all_features, device):
    """Build PyG data for a single node with its neighborhood."""
    ei = graph["edge_index"]
    src_np = ei[0].numpy()
    dst_np = ei[1].numpy()

    # Find 1-hop neighbors
    neighbors_mask = (src_np == node_id) | (dst_np == node_id)
    neighbor_src = ei[0][torch.from_numpy(neighbors_mask)]
    neighbor_dst = ei[1][torch.from_numpy(neighbors_mask)]

    neighbor_nodes = set([node_id])
    neighbor_nodes.update(neighbor_src.tolist())
    neighbor_nodes.update(neighbor_dst.tolist())
    neighbor_nodes = sorted(neighbor_nodes)

    id_to_new = {old: new for new, old in enumerate(neighbor_nodes)}
    mask2 = np.isin(src_np, neighbor_nodes) & np.isin(dst_np, neighbor_nodes)
    f_src = ei[0][torch.from_numpy(mask2)]
    f_dst = ei[1][torch.from_numpy(mask2)]
    new_src = torch.tensor([id_to_new[int(i)] for i in f_src], dtype=torch.long)
    new_dst = torch.tensor([id_to_new[int(i)] for i in f_dst], dtype=torch.long)

    return PyGData(
        x=all_features[neighbor_nodes],
        edge_index=torch.stack([new_src, new_dst]),
        y=graph["labels"][neighbor_nodes],
    ).to(device)


class GNNWithMLP(nn.Module):
    def __init__(self, in_dim=19, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.gnn1 = GINLayerV2(in_dim, hidden_dim, dropout)
        self.gnn2 = GINLayerV2(hidden_dim, hidden_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim + hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout / 2), nn.Linear(64, 1),
        )
        self.dropout_p = dropout

    def forward(self, data):
        h0 = data.x
        h1 = self.gnn1(h0, data.edge_index)
        h2 = self.gnn2(h1, data.edge_index)
        h_jk = torch.cat([h0, h1, h2], dim=-1)
        return self.classifier(h_jk).squeeze(-1)

    def forward_mc(self, data, T=10):
        self.train()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        probs_list = []
        with torch.no_grad():
            for _ in range(T):
                probs_list.append(torch.sigmoid(self.forward(data)))
        self.eval()
        probs = torch.stack(probs_list)
        mean_p = probs.mean(0)
        variance = probs.var(0, unbiased=False)
        eps = 1e-8
        entropy = -(mean_p * torch.log(mean_p + eps) + (1 - mean_p) * torch.log(1 - mean_p + eps))
        return mean_p, variance, entropy


from experiments.round3.train_gog_l1_v3 import (
    GNNWithMLP as TrainedGNNWithMLP,
    build_features_v3,
)


# ─────────────────────────────────────────────────────────────────────────────
# Load model (try v3 first, then v2, then v1)
# ─────────────────────────────────────────────────────────────────────────────

def load_best_model(device):
    """Load the best available checkpoint (v3 > v2 > v1)."""
    for seed in [7, 17, 27, 37, 47]:
        for prefix in ["l1v3_seed", "l1v2_seed", "l1_seed"]:
            p = CKPT_DIR / f"{prefix}{seed}_best.pt"
            if p.exists():
                ckpt = torch.load(p, map_location=device, weights_only=False)
                cfg = ckpt["model_config"]
                mc = ckpt.get("model_class", "Level1GNNDirect")
                if mc == "GNNWithMLP":
                    model = TrainedGNNWithMLP(
                        in_dim=cfg.get("in_dim", 18), hidden_dim=cfg.get("hidden_dim", 256),
                        dropout=cfg.get("dropout", 0.3),
                    )
                elif mc == "Level1GNNv2":
                    model = Level1GNNv2(
                        in_dim=cfg["in_dim"], hidden_dim=cfg["hidden_dim"],
                        num_layers=cfg["num_layers"], dropout=cfg["dropout"],
                    )
                else:
                    model = Level1GNNDirect(
                        in_dim=cfg["in_dim"], hidden_dim=cfg["hidden_dim"],
                        num_layers=cfg["num_layers"], dropout=cfg["dropout"],
                    )
                model.load_state_dict(ckpt["model_state_dict"])
                model = model.to(device)
                model.eval()
                log.info(f"  Loaded: {p.relative_to(ROOT)} ({mc})")
                return model, ckpt, p
    raise FileNotFoundError("No checkpoint found. Run training first.")



# ─────────────────────────────────────────────────────────────────────────────
# GraphRAG latency
# ─────────────────────────────────────────────────────────────────────────────

def measure_graphrag_latency(test_ids, contexts):
    """Measure GraphRAG retrieval + risk extraction latency."""
    try:
        from graphrag.local_kb import LocalKnowledgeBase
        from graphrag.retriever import GraphRAGRetriever, RetrieverConfig
        from graphrag.risk_extractor import RiskExtractor

        kb = LocalKnowledgeBase()
        config = RetrieverConfig(top_k=5, graph_hops=1)
        retriever = GraphRAGRetriever(kb=kb, config=config)
        extractor = RiskExtractor()

        sample = [nid for nid in test_ids if nid in contexts][:50]
        latencies = []
        for nid in sample:
            t0 = time.perf_counter()
            evidence = retriever.retrieve(contexts[nid])
            _ = extractor.extract(evidence)
            latencies.append((time.perf_counter() - t0) * 1000)
        return float(np.mean(latencies)), float(np.percentile(latencies, 95))
    except Exception as e:
        log.warning(f"GraphRAG latency unavailable: {e}")
        return None, None



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=100)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load data
    graph = torch.load(DATA_DIR / "polygon_hybrid_graph.pt", map_location="cpu", weights_only=False)
    train_ids = [int(x.strip()) for x in open(DATA_DIR / "train_ids.txt") if x.strip()]
    test_ids = [int(x.strip()) for x in open(DATA_DIR / "test_ids.txt") if x.strip()]
    train_labels = graph["labels"][train_ids]
    all_features = build_features_v3(graph, train_ids, train_labels)

    # Load contexts
    contexts = {}
    ctx_path = DATA_DIR / "contexts.jsonl"
    if ctx_path.exists():
        with open(ctx_path) as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    nid = int(c["event_id"].split("_")[1])
                    contexts[nid] = c.get("context_text", "")

    # Load model
    model, ckpt, checkpoint_path = load_best_model(device)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    from graphrag.local_kb import LocalKnowledgeBase
    from graphrag.retriever import GraphRAGRetriever, RetrieverConfig
    from graphrag.risk_extractor import RiskExtractor
    from graphrag.risk_encoder import RiskEncoder

    retriever = GraphRAGRetriever(
        LocalKnowledgeBase(), RetrieverConfig(top_k=5, graph_hops=1)
    )
    extractor = RiskExtractor()
    risk_encoder = RiskEncoder.from_config({}).to(device).eval()

    T_values = [1, 5, 10, 20, 30]
    results = []

    for T in T_values:
        log.info(f"\n=== T={T} ===")
        latencies = []

        # Exclude CUDA/kernel and Python first-call initialization from the
        # measured distribution.
        for warmup_id in test_ids[:10]:
            warmup_data = build_single_node_data(graph, warmup_id, all_features, device)
            model.forward_mc(warmup_data, T=T)
            warmup_text = contexts.get(warmup_id, "")
            warmup_evidence = retriever.retrieve(warmup_text) if warmup_text else []
            warmup_risk = extractor.extract(
                warmup_evidence, event_id=f"tx_{warmup_id:06d}"
            )
            risk_encoder.encode_risk_dict_batch([warmup_risk], device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Controlled single-event replay. The synthetic-time provenance is
        # recorded in every row and is rejected by the paper-ready gate.
        sample_ids = test_ids[:args.sample_count]

        for nid in sample_ids:
            t_start = time.perf_counter()

            # Step 1: Graph neighborhood retrieval
            t1 = time.perf_counter()
            node_data = build_single_node_data(graph, nid, all_features, device)
            t_graph_ms = (time.perf_counter() - t1) * 1000

            # Step 2: GNN forward (T MC passes)
            t2 = time.perf_counter()
            mean_p, var, ent = model.forward_mc(node_data, T=T)
            t_gnn_ms = (time.perf_counter() - t2) * 1000

            # Step 3: GraphRAG retrieval and risk extraction
            t3 = time.perf_counter()
            context_text = contexts.get(nid, "")
            evidence = retriever.retrieve(context_text) if context_text else []
            risk = extractor.extract(evidence, event_id=f"tx_{nid:06d}")
            t_graphrag_ms = (time.perf_counter() - t3) * 1000

            # Step 4: neural risk encoding
            t_risk = time.perf_counter()
            with torch.no_grad():
                _, encoded_risk = risk_encoder.encode_risk_dict_batch([risk], device=device)
            p_risk = float(encoded_risk.item())
            t_risk_encoder_ms = (time.perf_counter() - t_risk) * 1000

            # Step 5: uncertainty fusion
            t4 = time.perf_counter()
            p_gnn = float(mean_p[node_data.y.shape[0] // 2].item()) if mean_p.numel() > 0 else 0.5
            u_mc = float(var.mean().item())
            beta = 1.0 / (1.0 + np.exp(-10 * (u_mc - 0.01)))
            final_score = (1 - beta) * p_gnn + beta * p_risk
            t_fusion_ms = (time.perf_counter() - t4) * 1000

            # Step 6: event serialization
            t5 = time.perf_counter()
            json.dumps({
                "event_id": f"tx_{nid:06d}",
                "p_gnn": p_gnn,
                "u_mc": u_mc,
                "p_risk": p_risk,
                "score": final_score,
            }, separators=(",", ":"))
            t_serialization_ms = (time.perf_counter() - t5) * 1000

            t_total_ms = (time.perf_counter() - t_start) * 1000
            latencies.append({
                "total_ms": t_total_ms,
                "graph_ms": t_graph_ms,
                "gnn_ms": t_gnn_ms,
                "graphrag_ms": t_graphrag_ms,
                "risk_encoder_ms": t_risk_encoder_ms,
                "fusion_ms": t_fusion_ms,
                "serialization_ms": t_serialization_ms,
            })

        total_ms_arr = np.array([l["total_ms"] for l in latencies])
        gnn_ms_arr = np.array([l["gnn_ms"] for l in latencies])
        graphrag_ms_arr = np.array([l["graphrag_ms"] for l in latencies])
        risk_encoder_ms_arr = np.array([l["risk_encoder_ms"] for l in latencies])
        fusion_ms_arr = np.array([l["fusion_ms"] for l in latencies])
        serialization_ms_arr = np.array([l["serialization_ms"] for l in latencies])

        row = {
            "T": T,
            "gnn_source": "real_checkpoint",
            "split_type": "synthetic_time_ordered",
            "paper_eligible": False,
            "risk_encoder_source": "untrained_controlled_latency_only",
            "checkpoint_sha256": checkpoint_sha256,
            "n_events": len(sample_ids),
            "mean_total_ms": round(float(np.mean(total_ms_arr)), 4),
            "median_total_ms": round(float(np.median(total_ms_arr)), 4),
            "p95_total_ms": round(float(np.percentile(total_ms_arr, 95)), 4),
            "p99_total_ms": round(float(np.percentile(total_ms_arr, 99)), 4),
            "events_per_sec": round(1000.0 / float(np.mean(total_ms_arr)), 2),
            "mean_gnn_ms": round(float(np.mean(gnn_ms_arr)), 4),
            "p95_gnn_ms": round(float(np.percentile(gnn_ms_arr, 95)), 4),
            "mean_graphrag_ms": round(float(np.mean(graphrag_ms_arr)), 4),
            "mean_risk_encoder_ms": round(float(np.mean(risk_encoder_ms_arr)), 4),
            "mean_fusion_ms": round(float(np.mean(fusion_ms_arr)), 4),
            "mean_serialization_ms": round(float(np.mean(serialization_ms_arr)), 4),
        }

        # GPU memory
        if torch.cuda.is_available():
            row["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 2)
            torch.cuda.reset_peak_memory_stats()
        else:
            row["peak_gpu_mb"] = 0.0

        results.append(row)
        log.info(f"  Total: mean={row['mean_total_ms']:.2f}ms, "
                 f"p95={row['p95_total_ms']:.2f}ms, "
                 f"events/sec={row['events_per_sec']:.1f}")

    # GraphRAG latency
    log.info("\n=== GraphRAG Latency ===")
    gr_mean, gr_p95 = measure_graphrag_latency(test_ids, contexts)
    if gr_mean is not None:
        log.info(f"  GraphRAG mean={gr_mean:.2f}ms, p95={gr_p95:.2f}ms")
        for r in results:
            r["graphrag_mean_ms"] = round(gr_mean, 4)
            r["graphrag_p95_ms"] = round(gr_p95, 4)

    # Write CSV
    lat_path = RESULTS_DIR / "real_e2e_latency.csv"
    if results:
        with open(lat_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        log.info(f"\nWritten: {lat_path.relative_to(ROOT)}")

    log.info("\n=== Phase J Complete ===")


if __name__ == "__main__":
    main()
