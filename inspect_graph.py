"""Inspect GoG polygon_hybrid_graph.pt"""
import torch

g = torch.load(
    "data/benchmark/gog_microrag_stream_v1/polygon_hybrid_graph.pt",
    map_location="cpu",
    weights_only=False,
)
print("type:", type(g))
print(g)
if hasattr(g, "x"):
    print("x shape:", g.x.shape)
if hasattr(g, "edge_index"):
    print("edge_index shape:", g.edge_index.shape)
if hasattr(g, "y"):
    print("y shape:", g.y.shape, "fraud count:", int(g.y.sum().item()))
for attr in ["timestamp", "edge_time", "time", "node_time"]:
    if hasattr(g, attr):
        t = getattr(g, attr)
        print(f"  has {attr}: shape={t.shape if hasattr(t,'shape') else 'N/A'}, dtype={t.dtype if hasattr(t,'dtype') else 'N/A'}")
print("all attrs:", [a for a in dir(g) if not a.startswith("_")])
