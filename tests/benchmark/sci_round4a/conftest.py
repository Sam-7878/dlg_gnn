import pytest
import torch
from torch_geometric.utils import to_dense_adj


@pytest.fixture
def reconstruction_case():
    torch.manual_seed(17)
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3, 4, 4], [0, 1, 2, 1, 4, 3, 4]], dtype=torch.long
    )
    z = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    adjacency = to_dense_adj(edge_index, max_num_nodes=5)[0].to(torch.float64)
    return edge_index, z, adjacency
