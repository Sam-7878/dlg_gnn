import torch
from torch_geometric.data import Data

import _round3_bootstrap  # noqa: F401

from experiments.round3.train_gog_l1_v3 import GNNWithMLP


def _data():
    torch.manual_seed(3)
    return Data(
        x=torch.randn(8, 4),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]),
    )


def test_mc_dropout_has_nonzero_population_variance():
    model = GNNWithMLP(in_dim=4, hidden_dim=8, dropout=0.5)
    _, variance, _ = model.forward_mc(_data(), T=10)
    assert torch.isfinite(variance).all()
    assert torch.count_nonzero(variance).item() > 0


def test_t1_population_variance_is_finite_zero():
    model = GNNWithMLP(in_dim=4, hidden_dim=8, dropout=0.5)
    _, variance, _ = model.forward_mc(_data(), T=1)
    assert torch.isfinite(variance).all()
    assert torch.equal(variance, torch.zeros_like(variance))
