#!/usr/bin/env python3
"""
Test feature scaling on DOMINANT for theia_graph.pt
"""
import torch
import numpy as np
from gog_fraud.models.pygod.stable_reconstruction import DOMINANT

def main():
    print("Loading theia_graph.pt...")
    data = torch.load("outputs/sci_defense_extension_real/graphs/theia_graph.pt", weights_only=False)
    print("Original x stats: min=", data.x.min().item(), "max=", data.x.max().item())

    # Apply log1p on numerical count/duration columns (indices 10 to 19)
    x_new = data.x.clone()
    x_new[:, 10:] = torch.log1p(x_new[:, 10:])
    
    # Standardize all columns with non-zero std
    means = x_new.mean(dim=0, keepdim=True)
    stds = x_new.std(dim=0, keepdim=True)
    stds[stds == 0] = 1.0
    x_new = (x_new - means) / stds

    print("Normalized x stats: min=", x_new.min().item(), "max=", x_new.max().item(), "mean=", x_new.mean().item(), "std=", x_new.std().item())
    data.x = x_new

    # Test DOMINANT
    print("Instantiating DOMINANT...")
    model = DOMINANT(hid_dim=32, num_layers=2, epoch=2, lr=0.005, weight_decay=0.0)
    print("Fitting model...")
    model.fit(data)
    scores = model.decision_score_
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores).reshape(-1)
    print("Scores stats: min=", scores.min(), "max=", scores.max(), "mean=", scores.mean())
    print("Finite count:", np.isfinite(scores).sum(), "total:", len(scores))
    assert np.isfinite(scores).all(), "Scores must be finite!"
    print("DOMINANT PASSED WITH FINITE SCORES!")

if __name__ == "__main__":
    main()
