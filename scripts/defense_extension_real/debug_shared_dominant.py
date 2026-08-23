#!/usr/bin/env python3
"""
Test SharedDOMINANT and SharedDLGBase on theia_graph.pt
"""
import torch
import numpy as np
from gog_fraud.models.pygod.shared_reconstruction import SharedDOMINANT, SharedDLGBase

def main():
    print("Loading theia_graph.pt...")
    data = torch.load("outputs/sci_defense_extension_real/graphs/theia_graph.pt", weights_only=False)
    print("Original x stats: min=", data.x.min().item(), "max=", data.x.max().item())

    # Apply log1p on numerical count/duration columns (indices 10 to 19)
    x_new = data.x.clone().float()
    large_cols = x_new.max(dim=0).values > 10.0
    x_new[:, large_cols] = torch.log1p(torch.clamp(x_new[:, large_cols], min=0.0))
    
    # Standardize
    means = x_new.mean(dim=0, keepdim=True)
    stds = x_new.std(dim=0, keepdim=True)
    stds[stds == 0] = 1.0
    x_new = (x_new - means) / stds

    data.x = x_new
    print("Normalized x stats: min=", x_new.min().item(), "max=", x_new.max().item())

    # Test SharedDOMINANT
    print("Instantiating SharedDOMINANT...")
    model = SharedDOMINANT(
        hid_dim=32, num_layers=2, epoch=1, lr=0.005, weight_decay=0.0,
        message_backend="sparse_fused", reconstruction_backend="exact_sparse",
        score_chunk_size=8192, gpu=-1, verbose=1,
    )
    print("Fitting SharedDOMINANT for 1 epoch on CPU...")
    model.fit(data)
    scores = model.decision_score_
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores).reshape(-1)
    print("SharedDOMINANT scores: min=", scores.min(), "max=", scores.max(), "mean=", scores.mean())
    print("Finite count:", np.isfinite(scores).sum(), "total:", len(scores))
    assert np.isfinite(scores).all(), "Scores must be finite!"
    print("SharedDOMINANT PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
