import copy

import pytest
import torch
from torch_geometric.nn import GCN

from gog_fraud.models.pygod.sparse_message import SparseFusedGCN, normalized_sparse_adjt


@pytest.fixture
def message_case():
    torch.manual_seed(101)
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7],
         [0, 1, 2, 1, 3, 4, 3, 6, 7, 6, 5, 7]], dtype=torch.long,
    )
    x = torch.randn(8, 5, dtype=torch.float64)
    return x, edge_index


def model_pair(x, edge_index, layers, edge_weight=None):
    reference = GCN(
        in_channels=x.shape[1], hidden_channels=7, out_channels=4,
        num_layers=layers, dropout=0,
    ).double()
    fused = SparseFusedGCN(
        in_channels=x.shape[1], hidden_channels=7, out_channels=4,
        num_layers=layers, dropout=0,
    ).double()
    fused.load_state_dict(copy.deepcopy(reference.state_dict()))
    adj_t = normalized_sparse_adjt(
        edge_index, x.shape[0], edge_weight=edge_weight, dtype=x.dtype,
    )
    return reference, fused, adj_t
