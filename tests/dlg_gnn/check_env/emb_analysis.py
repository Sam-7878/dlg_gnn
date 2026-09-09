"""Quick embedding analysis."""
import torch, numpy as np
g = torch.load('data/benchmark/gog_microrag_stream_v1/polygon_hybrid_graph.pt', map_location='cpu', weights_only=False)
emb = g['embeddings']
labels = g['labels']
print('Dim-by-dim stats:')
for i in range(8):
    col = emb[:, i]
    nz = (col != 0).sum().item()
    unique_vals = col.unique()
    print(f'  dim {i}: min={col.min():.3f} max={col.max():.3f} '
          f'nonzero={nz}/{len(col)} unique_count={len(unique_vals)} sample_uniq={unique_vals[:5].tolist()}')

# Check fraud feature distribution
fraud_mask = labels == 1
print(f'\nFraud embedding stats (n={fraud_mask.sum()}):')
fraud_emb = emb[fraud_mask]
benign_emb = emb[~fraud_mask]
for i in range(8):
    fd = fraud_emb[:, i]
    bd = benign_emb[:, i]
    print(f'  dim {i}: fraud_mean={fd.mean():.4f} benign_mean={bd.mean():.4f} '
          f'fraud_std={fd.std():.4f} benign_std={bd.std():.4f}')

# Check metadata.json
import json
with open('data/benchmark/gog_microrag_stream_v1/metadata.json') as f:
    meta = json.load(f)
print('\nmetadata.json:')
print(json.dumps(meta, indent=2))
