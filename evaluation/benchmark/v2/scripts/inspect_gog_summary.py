import os
import torch
import numpy as np

for chain in ["polygon", "bsc", "ethereum"]:
    path = f"/mnt/d/_Work/_data/GoG/{chain}/{chain}_level2_graph.pt"
    if not os.path.exists(path):
        path = f"/mnt/d/_Work/_data/GoG/{chain}/{chain}_hybrid_graph.pt"

    data = torch.load(path, map_location="cpu", weights_only=False)
    print("=" * 50)
    print(f"Chain: {chain}")
    print("Keys:", list(data.keys()))
    num_nodes = data.get("num_nodes")
    edge_index = data.get("edge_index")
    labels = data.get("labels")
    embeddings = data.get("embeddings")
    
    print(f"  num_nodes: {num_nodes}")
    if edge_index is not None:
        print(f"  edge_index shape: {edge_index.shape}, dtype: {edge_index.dtype}")
    if embeddings is not None:
        print(f"  embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            y = labels.view(-1)
        else:
            y = torch.tensor(labels).view(-1)
        pos = (y == 1).sum().item()
        print(f"  labels shape: {y.shape}, pos: {pos} ({pos/len(y)*100:.2f}%)")
