import torch

for chain in ["polygon", "bsc", "ethereum"]:
    path = f"/mnt/d/_Work/_data/GoG/{chain}/{chain}_hybrid_graph.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Chain: {chain}")
    print(f"  Type: {type(data)}")
    print(f"  Data: {data}")
    if hasattr(data, "x") and data.x is not None:
        print(f"  Nodes: {data.x.shape[0]}, Dim: {data.x.shape[1]}")
    if hasattr(data, "edge_index") and data.edge_index is not None:
        print(f"  Edges: {data.edge_index.shape[1]}")
    if hasattr(data, "y") and data.y is not None:
        y = data.y.view(-1)
        pos = (y == 1).sum().item()
        total = len(y)
        print(f"  Labels: total={total}, pos={pos} ({pos/total*100:.2f}%)")
