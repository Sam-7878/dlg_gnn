"""Deep inspection of all key checkpoint and data files for Round 3 audit."""
import torch
import os

print("=" * 70)
print("1. polygon_hybrid_graph.pt")
print("=" * 70)
g = torch.load(
    "data/benchmark/gog_microrag_stream_v1/polygon_hybrid_graph.pt",
    map_location="cpu", weights_only=False,
)
print("type:", type(g))
if isinstance(g, dict):
    for k, v in g.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            if k == "labels":
                print(f"    fraud={int(v.sum())}, total={len(v)}, ratio={float(v.float().mean()):.4f}")
        else:
            print(f"  {k}: {type(v)}")

print()
print("=" * 70)
print("2. l1_model_weights_polygon.pt (benchmark_mc_streaming)")
print("=" * 70)
ckpt = torch.load(
    "results/benchmark_mc_streaming/l1_model_weights_polygon.pt",
    map_location="cpu", weights_only=False,
)
print("type:", type(ckpt))
if isinstance(ckpt, (dict,)):
    all_keys = list(ckpt.keys())
    print(f"  num keys: {len(all_keys)}")
    print("  first 5 keys:", all_keys[:5])
    print("  last 5 keys:", all_keys[-5:])
    for k in all_keys[:3]:
        v = ckpt[k]
        if hasattr(v, "shape"):
            print(f"  [{k}] shape={v.shape}")

print()
print("=" * 70)
print("3. l2_model_weights_polygon.pt (benchmark_mc_streaming)")
print("=" * 70)
ckpt2 = torch.load(
    "results/benchmark_mc_streaming/l2_model_weights_polygon.pt",
    map_location="cpu", weights_only=False,
)
if isinstance(ckpt2, dict):
    print(f"  num keys: {len(ckpt2)}")
    for k in list(ckpt2.keys())[:5]:
        v = ckpt2[k]
        if hasattr(v, "shape"):
            print(f"  [{k}] shape={v.shape}")

print()
print("=" * 70)
print("4. sci_round5_final checkpoint (first largest .pt)")
print("=" * 70)
sc5_dir = "outputs/sci_round5_final/checkpoints"
big_files = sorted(
    [(os.path.getsize(os.path.join(sc5_dir, f)), f)
     for f in os.listdir(sc5_dir) if f.endswith(".pt") and ".progress" not in f],
    reverse=True
)[:2]
for sz, fn in big_files:
    fpath = os.path.join(sc5_dir, fn)
    try:
        ckpt5 = torch.load(fpath, map_location="cpu", weights_only=False)
        print(f"\n  {fn} ({sz/1e6:.1f} MB): type={type(ckpt5)}")
        if isinstance(ckpt5, dict):
            print(f"    top-level keys: {list(ckpt5.keys())[:10]}")
        break  # Just one
    except Exception as e:
        print(f"  {fn}: ERROR {e}")

print()
print("=" * 70)
print("5. train_ids.txt sample")
print("=" * 70)
with open("data/benchmark/gog_microrag_stream_v1/train_ids.txt") as f:
    ids = [l.strip() for l in f.readlines() if l.strip()]
    print(f"  train: {len(ids)} ids, first={ids[0]}, last={ids[-1]}")
with open("data/benchmark/gog_microrag_stream_v1/valid_ids.txt") as f:
    ids = [l.strip() for l in f.readlines() if l.strip()]
    print(f"  valid: {len(ids)} ids, first={ids[0]}, last={ids[-1]}")
with open("data/benchmark/gog_microrag_stream_v1/test_ids.txt") as f:
    ids = [l.strip() for l in f.readlines() if l.strip()]
    print(f"  test: {len(ids)} ids, first={ids[0]}, last={ids[-1]}")
