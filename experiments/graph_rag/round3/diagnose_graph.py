"""
Diagnose GoG-MicroRAG-Stream-v1 polygon_hybrid_graph.pt
- Are embeddings all zeros?
- What features are available?
- Can we build alternative node features from edge structure?
"""
import torch
import numpy as np
from pathlib import Path

ROOT = Path("/mnt/d/_Work/goat_bank/dlg_gnn")
DATA_DIR = ROOT / "data" / "benchmark" / "gog_microrag_stream_v1"

graph = torch.load(DATA_DIR / "polygon_hybrid_graph.pt", map_location="cpu", weights_only=False)
emb = graph["embeddings"]   # [2303, 8]
ei = graph["edge_index"]    # [2, 18411]
labels = graph["labels"]    # [2303]

print("=== Embeddings analysis ===")
print(f"  shape: {emb.shape}, dtype: {emb.dtype}")
print(f"  min: {emb.min():.6f}, max: {emb.max():.6f}, mean: {emb.mean():.6f}, std: {emb.std():.6f}")

zero_rows = (emb.abs().sum(dim=1) == 0).sum().item()
print(f"  zero rows (all-zero embedding): {zero_rows}/{emb.shape[0]} = {zero_rows/emb.shape[0]*100:.1f}%")

nonzero_rows = emb.shape[0] - zero_rows
print(f"  nonzero rows: {nonzero_rows}")

# Show a few nonzero rows
nonzero_idx = (emb.abs().sum(dim=1) > 0).nonzero().flatten()
if len(nonzero_idx) > 0:
    print(f"  First 5 nonzero row indices: {nonzero_idx[:5].tolist()}")
    print(f"  Sample nonzero rows:\n{emb[nonzero_idx[:5]]}")

print("\n=== Edge analysis ===")
print(f"  edge_index shape: {ei.shape}")
print(f"  num unique source nodes: {ei[0].unique().numel()}")
print(f"  num unique dest nodes: {ei[1].unique().numel()}")

# Node degree
N = emb.shape[0]
degree = torch.zeros(N, dtype=torch.long)
degree.index_add_(0, ei[0], torch.ones(ei.shape[1], dtype=torch.long))
degree.index_add_(0, ei[1], torch.ones(ei.shape[1], dtype=torch.long))
print(f"  degree: min={degree.min()}, max={degree.max()}, mean={degree.float().mean():.2f}")

print("\n=== Labels ===")
print(f"  fraud: {labels.sum()}/{len(labels)} = {float(labels.float().mean())*100:.2f}%")
per_split_ids = {
    "train": [int(x.strip()) for x in open(DATA_DIR/"train_ids.txt") if x.strip()],
    "valid": [int(x.strip()) for x in open(DATA_DIR/"valid_ids.txt") if x.strip()],
    "test": [int(x.strip()) for x in open(DATA_DIR/"test_ids.txt") if x.strip()],
}
for split, ids in per_split_ids.items():
    y = labels[ids]
    print(f"  {split}: {y.sum()}/{len(ids)} fraud, "
          f"zero-emb: {(emb[ids].abs().sum(dim=1)==0).sum()}/{len(ids)}")

print("\n=== All graph keys ===")
for k, v in graph.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
              f"min={float(v.float().min()):.4f}, max={float(v.float().max()):.4f}")
    else:
        print(f"  {k}: {type(v)}")

print("\n=== Conclusion ===")
if zero_rows > 0.5 * N:
    print("  PROBLEM: >50% of embeddings are zero.")
    print("  Solution: Build structural node features (degree, clustering coeff, etc.)")
    print("  or use Graph-structural-only features (degree + in/out degree + random walk)")
