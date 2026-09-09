import torch

from conftest import model_pair


def test_sparse_gcn_duplicate_edge_equivalence():
    x = torch.randn(4, 3, dtype=torch.float64)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 2], [1, 1, 1, 2, 3, 3]])
    reference, fused, adj_t = model_pair(x, edge_index, 2)
    torch.testing.assert_close(fused(x, adj_t), reference(x, edge_index), rtol=1e-10, atol=1e-11)
