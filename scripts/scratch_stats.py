import torch
import os

chains = ["polygon", "bsc", "ethereum"]

for chain in chains:
    path = f"/mnt/d/_Work/_data/GoG/{chain}/{chain}_hybrid_graph.pt"
    if os.path.exists(path):
        print(f"=== {chain.upper()} ===")
        try:
            data = torch.load(path)
            print("Type:", type(data))
            if isinstance(data, dict):
                print("Keys:", list(data.keys()))
                labels = data['labels']
                edge_index = data['edge_index']
                num_nodes = len(data['idx_to_contract'])
                num_edges = edge_index.shape[1]
                
                # Check for bidirectional edges
                edge_set = set()
                for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                    edge_set.add((min(u, v), max(u, v)))
                num_undirected_edges = len(edge_set)
                
                print("Num Nodes:", num_nodes)
                print("Num Edges (directed/pairs):", num_edges)
                print("Num Edges (undirected):", num_undirected_edges)
                
                # Class distribution
                unique, counts = torch.unique(labels, return_counts=True)
                dist = dict(zip(unique.tolist(), counts.tolist()))
                print("Label distribution:", dist)
                total = len(labels)
                for label, count in dist.items():
                    print(f"  Class {label}: {count} ({count/total*100:.2f}%)")
            else:
                if hasattr(data, "num_nodes"):
                    print("num_nodes:", data.num_nodes)
                if hasattr(data, "num_edges"):
                    print("num_edges:", data.num_edges)
                if hasattr(data, "edge_index"):
                    print("edge_index shape:", data.edge_index.shape)
                if hasattr(data, "y"):
                    print("y shape:", data.y.shape)
                    unique, counts = torch.unique(data.y, return_counts=True)
                    dist = dict(zip(unique.tolist(), counts.tolist()))
                    print("y class distribution:", dist)
        except Exception as e:
            print("Error loading:", e)
    else:
        print(f"Path does not exist: {path}")