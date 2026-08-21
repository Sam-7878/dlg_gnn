"""Test zero label leakage in DARPA-TC-THEIA dataset."""
from pathlib import Path
import numpy as np
import torch

from gog_fraud.extensions.defense.defense_registry import load_defense_dataset


def test_theia_no_label_leakage():
    data = load_defense_dataset("DARPA-TC-THEIA")
    x = data.x.detach().cpu().numpy().astype(np.float64)
    y = data.y.detach().cpu().numpy().astype(np.float64)

    assert torch.isfinite(data.x).all(), "THEIA feature matrix contains NaN or Inf"
    assert len(x) == len(y), "Mismatch between feature rows and label length"

    # Ensure no individual feature column is perfectly correlated with y (r > 0.99)
    for col_idx in range(x.shape[1]):
        feat = x[:, col_idx]
        if np.std(feat) > 1e-8 and np.std(y) > 1e-8:
            corr = abs(float(np.corrcoef(feat, y)[0, 1]))
            assert corr < 0.95, f"Feature {col_idx} suspiciously correlated ({corr:.4f}) with label y"


def test_theia_graph_structure():
    data = load_defense_dataset("DARPA-TC-THEIA")
    n = data.num_nodes
    e = data.edge_index

    assert e.size(0) == 2
    assert e.min() >= 0
    assert e.max() < n
    assert int((data.y == 1).sum()) > 0, "No positive nodes in THEIA"
    assert int((data.y == 0).sum()) > 0, "No negative nodes in THEIA"
