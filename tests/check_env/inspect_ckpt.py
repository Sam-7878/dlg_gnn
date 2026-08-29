"""Quick inspection of checkpoint files."""
import torch
import sys

paths = [
    "results/benchmark_mc_streaming/l1_model_weights_polygon.pt",
    "results/benchmark_mc_streaming/l2_model_weights_polygon.pt",
    "results/benchmark_ngnn_mc_streaming/l1_model_weights_polygon.pt",
    "results/benchmark_ngnn_mc_streaming/l2_model_weights_polygon.pt",
]

for p in paths:
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
        print(f"\n=== {p} ===")
        print(f"  type: {type(d)}")
        if isinstance(d, dict):
            print(f"  keys: {list(d.keys())[:15]}")
            # check if it's a state_dict or a model wrapper
            for k in list(d.keys())[:5]:
                v = d[k]
                print(f"    [{k}]: type={type(v).__name__}", end="")
                if hasattr(v, "shape"):
                    print(f", shape={v.shape}")
                else:
                    print(f", val={repr(v)[:60]}")
        else:
            print(f"  value repr: {repr(d)[:200]}")
    except Exception as e:
        print(f"  ERROR: {e}")
