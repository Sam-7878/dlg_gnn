# -*- coding: utf-8 -*-
"""
DLG-GNN Benchmark: Graph Topology and Homophily Calculator

Important: Dataset loading must match benchmark_8x10_pipeline.py exactly.
Datasets that use injected outliers in the benchmark must also inject
outliers here, because _inject_outliers() overwrites data.y entirely.
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
from torch_geometric.datasets import (
    Planetoid, Flickr, Reddit,
    EllipticBitcoinDataset, Yelp,
    Amazon, BitcoinOTC
)
from torch_geometric.utils import remove_self_loops
from pygod.generator import gen_contextual_outlier, gen_structural_outlier


def _inject_outliers(data, contextual_ratio=0.03, structural_ratio=0.03,
                     m_clique=10, k=50):
    """Inject synthetic contextual and structural outliers into a PyG Data
    object. Mirrors benchmark_8x10_pipeline._inject_outliers exactly.

    IMPORTANT: This completely **overwrites** data.y with the injected
    outlier labels (logical OR of contextual and structural outlier masks).
    """
    n_contextual = max(10, int(data.num_nodes * contextual_ratio))
    data, yc = gen_contextual_outlier(data, n=n_contextual, k=k, seed=42)

    n_clique = max(1, int((data.num_nodes * structural_ratio) / m_clique))
    data, ys = gen_structural_outlier(data, m=m_clique, n=n_clique, seed=42)

    data.y = torch.logical_or(yc, ys).long()
    return data


def _repackage_graph(data):
    """Repackage graph with self-loops, edge validation, and feature
    sanitization. Mirrors benchmark_8x10_pipeline._repackage_graph."""
    from torch_geometric.utils import add_self_loops, coalesce
    from torch_geometric.data import Data as PyGData

    num_nodes = data.num_nodes
    edge_index = data.edge_index.long()

    # Remove out-of-range edges
    if edge_index.numel() > 0:
        valid = ((edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                 & (edge_index[0] >= 0) & (edge_index[1] >= 0))
        if not valid.all():
            edge_index = edge_index[:, valid]

    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_index = coalesce(edge_index)

    x = data.x.float()
    nan_count = torch.isnan(x).sum().item() + torch.isinf(x).sum().item()
    if nan_count > 0:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    clean = PyGData(x=x, edge_index=edge_index, num_nodes=num_nodes)
    if hasattr(data, 'y') and data.y is not None:
        clean.y = data.y
    return clean

# Path Configuration
# Check if running under Windows or Linux and resolve data directories
DATA_ROOT = r"d:\_Work\_data\DLG"
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = "/mnt/d/_Work/_data/DLG"

OUTPUT_DIR = r"d:\_Work\goat_bank\dlg_gnn\docs\work_reports\36-domain_wise_ranking\metadata"
if not os.path.exists(r"d:\_Work\goat_bank\dlg_gnn"):
    OUTPUT_DIR = "docs/work_reports/36-domain_wise_ranking/metadata"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_edge_homophily(edge_index, labels):
    """Compute standard edge homophily score."""
    src = edge_index[0]
    dst = edge_index[1]
    
    # Filter labels to ensure valid non-negative index mapping
    valid = (labels[src] >= 0) & (labels[dst] >= 0)
    src_valid = src[valid]
    dst_valid = dst[valid]
    
    if src_valid.numel() == 0:
        return float("nan")
    
    matches = (labels[src_valid] == labels[dst_valid]).sum().item()
    return float(matches / src_valid.numel())

def compute_class_balanced_homophily(edge_index, labels, num_classes=2):
    """Compute class-balanced edge homophily to mitigate majority class bias."""
    src = edge_index[0]
    dst = edge_index[1]
    
    valid = (labels[src] >= 0) & (labels[dst] >= 0)
    src_valid = src[valid]
    dst_valid = dst[valid]
    
    if src_valid.numel() == 0:
        return float("nan")
        
    class_homophilies = []
    for c in range(num_classes):
        c_mask = (labels[src_valid] == c)
        c_src = src_valid[c_mask]
        c_dst = dst_valid[c_mask]
        
        if c_src.numel() == 0:
            continue
            
        matches = (labels[c_src] == labels[c_dst]).sum().item()
        class_homophilies.append(matches / c_src.numel())
        
    if len(class_homophilies) == 0:
        return float("nan")
    return float(np.mean(class_homophilies))

def load_elliptic():
    root = os.path.join(DATA_ROOT, "Elliptic")
    dataset = EllipticBitcoinDataset(root=root)
    data = dataset[0]
    if data.y.dim() > 1:
        data.y = data.y.squeeze(-1)
    # Remove unknown-label nodes
    known_mask = (data.y != 2)
    data = data.subgraph(known_mask)
    return data

def load_dgraphfin():
    npz_path = os.path.join(DATA_ROOT, "DGraphFin", "dgraphfin.npz")
    if not os.path.exists(npz_path):
        print(f"    [Warning] DGraphFin file not found at {npz_path}")
        return None
    
    loader = np.load(npz_path)
    x = torch.from_numpy(loader['x']).float()
    y = torch.from_numpy(loader['y']).long()
    edge_index = torch.from_numpy(loader['edge_index']).long().t().contiguous()
    
    if y.dim() > 1:
        y = y.squeeze(-1)
        
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, y=y)
    known_mask = (data.y == 0) | (data.y == 1)
    data = data.subgraph(known_mask)
    return data

def load_yelp():
    root = os.path.join(DATA_ROOT, "Yelp")
    dataset = Yelp(root=root)
    data = dataset[0]
    if data.y.dim() > 1:
        data.y = (data.y.sum(dim=-1) > 0).long()
    # Repackage graph (sanitize features, add self-loops) matching the pipeline
    data = _repackage_graph(data)
    # Inject outliers with the same ratios as the benchmark pipeline.
    # CRITICAL: _inject_outliers() overwrites data.y entirely, replacing
    # the incorrect 99.9% multi-label conversion with correct ~2% outlier labels.
    data = _inject_outliers(data, contextual_ratio=0.01, structural_ratio=0.01,
                            m_clique=8)
    return data

def load_amazon():
    root = os.path.join(DATA_ROOT, "Amazon")
    dataset = Amazon(root=root, name='Computers')
    data = dataset[0]
    data = _repackage_graph(data)
    # Amazon Computers has no anomaly labels; inject synthetic outliers
    # matching the benchmark pipeline (contextual=3%, structural=2%, m=8).
    data = _inject_outliers(data, contextual_ratio=0.03, structural_ratio=0.02,
                            m_clique=8)
    return data

def load_bitcoin_otc():
    root = os.path.join(DATA_ROOT, "BitcoinOTC")
    dataset = BitcoinOTC(root=root, edge_window_size=10)
    data = dataset[-1]

    from torch_geometric.utils import degree
    from torch_geometric.nn import Node2Vec

    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # ── 1. Node2Vec embedding (64-dim) ──
    print("    ↳ Training Node2Vec for BitcoinOTC...", end=" ", flush=True)
    n2v_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    node2vec = Node2Vec(
        edge_index,
        embedding_dim=64,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        num_nodes=num_nodes,
    ).to(n2v_device)

    loader = node2vec.loader(batch_size=256, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(list(node2vec.parameters()), lr=0.01)

    node2vec.train()
    for epoch in range(1, 101):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(n2v_device), neg_rw.to(n2v_device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    node2vec.eval()
    with torch.no_grad():
        n2v_emb = node2vec().detach().cpu()

    # ── 2. Hand-crafted structural features (3-dim) ──
    deg = degree(edge_index[0], num_nodes=num_nodes)
    log_deg = torch.log1p(deg)
    struct_feat = torch.stack([
        deg,
        log_deg,
        deg / (deg.max() + 1e-6),
    ], dim=-1).float()

    # ── 3. Concatenate: [Node2Vec(64) | struct(3)] → 67-dim ──
    data.x = torch.cat([n2v_emb, struct_feat], dim=-1).float()
    data.y = torch.zeros(num_nodes, dtype=torch.long)

    data = _repackage_graph(data)
    data = _inject_outliers(data, contextual_ratio=0.03, structural_ratio=0.03,
                            m_clique=8)
    return data

def load_flickr():
    root = os.path.join(DATA_ROOT, "Flickr")
    dataset = Flickr(root=root)
    data = dataset[0]
    data = _repackage_graph(data)
    data = _inject_outliers(data, contextual_ratio=0.02, structural_ratio=0.02,
                            m_clique=8)
    return data

def load_reddit():
    root = os.path.join(DATA_ROOT, "Reddit")
    dataset = Reddit(root=root)
    data = dataset[0]
    data = _repackage_graph(data)
    data = _inject_outliers(data, contextual_ratio=0.02, structural_ratio=0.01,
                            m_clique=10)
    return data

def load_planetoid(name):
    root = os.path.join(DATA_ROOT, name)
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    data = _inject_outliers(data, contextual_ratio=0.03, structural_ratio=0.03,
                            m_clique=8)
    return data

def main():
    loaders = {
        "Elliptic": load_elliptic,
        "DGraphFin": load_dgraphfin,
        "Yelp": load_yelp,
        "Amazon": load_amazon,
        "BitcoinOTC": load_bitcoin_otc,
        "Flickr": load_flickr,
        "Reddit": load_reddit,
        "Cora": lambda: load_planetoid("Cora"),
        "CiteSeer": lambda: load_planetoid("CiteSeer"),
        "PubMed": lambda: load_planetoid("PubMed")
    }
    
    records = []
    
    for name, loader in loaders.items():
        print(f"Processing dataset: {name}...")
        try:
            data = loader()
            if data is None:
                raise ValueError("Dataset loader returned None")
                
            num_nodes = data.num_nodes
            # Clean self-loops to calculate topology features accurately
            edge_index_clean, _ = remove_self_loops(data.edge_index)
            num_edges = edge_index_clean.size(1)
            
            anomaly_ratio = float((data.y == 1).sum().item() / num_nodes)
            edge_homophily = compute_edge_homophily(edge_index_clean, data.y)
            class_balanced_homophily = compute_class_balanced_homophily(edge_index_clean, data.y)
            
            avg_degree = float(num_edges / num_nodes)
            density = float(num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0.0
            
            records.append({
                "Dataset": name,
                "Nodes": num_nodes,
                "Edges": num_edges,
                "AnomalyRatio": round(anomaly_ratio, 6),
                "EdgeHomophily": round(edge_homophily, 6),
                "ClassBalancedHomophily": round(class_balanced_homophily, 6),
                "AvgDegree": round(avg_degree, 4),
                "Density": round(density, 8)
            })
            print(f"  -> Nodes: {num_nodes:,}, Edges: {num_edges:,}, Homophily: {edge_homophily:.4f}, AnomalyRatio: {anomaly_ratio:.4f}")
        except Exception as e:
            print(f"  [ERROR] Failed to compute topology for {name}: {e}")
            import traceback
            traceback.print_exc()
            
    # Write topology metrics to metadata folder
    df = pd.DataFrame(records)
    output_path = os.path.join(OUTPUT_DIR, "dataset_topology_metadata.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved dataset topology metadata to: {output_path}")

if __name__ == "__main__":
    main()
