import copy

import torch
from torch_geometric.utils import to_dense_adj

from gog_fraud.models.pygod.exact_reconstruction import exact_double_reconstruction_score
from gog_fraud.models.pygod.shared_reconstruction import (
    ExactDLGBase,
    ExactDOMINANTBase,
)
from gog_fraud.models.pygod.stable_reconstruction import (
    StableDOMINANTBase,
    stable_reconstruction_score,
)
from gog_fraud.models.pygod.dlg_base import DLGBase


def _graph():
    torch.manual_seed(31)
    x = torch.randn(7, 4, dtype=torch.float64)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 5, 6, 6], [1, 0, 2, 1, 4, 3, 6, 5, 6]],
        dtype=torch.long,
    )
    adjacency = to_dense_adj(edge_index, max_num_nodes=7)[0].to(torch.float64)
    return x, edge_index, adjacency


def _train_pair(dense, exact, x, edge_index, adjacency, steps=3):
    exact.load_state_dict(copy.deepcopy(dense.state_dict()))
    dense_opt = torch.optim.Adam(dense.parameters(), lr=0.003)
    exact_opt = torch.optim.Adam(exact.parameters(), lr=0.003)
    dense_history, exact_history = [], []
    for _ in range(steps):
        dense_opt.zero_grad()
        x_hat, adjacency_hat = dense(x, edge_index)
        dense_score = stable_reconstruction_score(x, x_hat, adjacency, adjacency_hat)
        dense_loss = dense_score.mean()
        dense_loss.backward()
        dense_opt.step()
        dense_history.append(dense_loss.detach())

        exact_opt.zero_grad()
        x_hat, z_structure = exact(x, edge_index)
        exact_score = exact_double_reconstruction_score(
            x, x_hat, z_structure, edge_index, backend="exact_sparse"
        )
        exact_loss = exact_score.mean()
        exact_loss.backward()
        exact_opt.step()
        exact_history.append(exact_loss.detach())

    torch.testing.assert_close(torch.stack(exact_history), torch.stack(dense_history), rtol=1e-9, atol=1e-10)
    for dense_parameter, exact_parameter in zip(dense.parameters(), exact.parameters()):
        torch.testing.assert_close(exact_parameter, dense_parameter, rtol=1e-8, atol=1e-9)


def test_dominant_shared_training_matches_dense_reference():
    x, edge_index, adjacency = _graph()
    dense = StableDOMINANTBase(in_dim=4, hid_dim=5, num_layers=4).double()
    exact = ExactDOMINANTBase(in_dim=4, hid_dim=5, num_layers=4).double()
    _train_pair(dense, exact, x, edge_index, adjacency)


def test_dlg_base_shared_training_matches_dense_reference():
    x, edge_index, adjacency = _graph()
    dense = DLGBase(in_dim=4, hid_dim=5, num_layers=4).double()
    exact = ExactDLGBase(in_dim=4, hid_dim=5, num_layers=4).double()
    _train_pair(dense, exact, x, edge_index, adjacency)
