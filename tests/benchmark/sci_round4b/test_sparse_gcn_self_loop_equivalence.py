import torch

from conftest import model_pair


def test_sparse_gcn_existing_weighted_self_loops_equivalence():
    x = torch.randn(4, 3, dtype=torch.float64)
    edge_index = torch.tensor([[0, 0, 1, 2, 3, 3], [0, 1, 1, 3, 2, 3]])
    edge_weight = torch.tensor([2., .5, 3., 1.5, .75, 4.], dtype=torch.float64)
    reference, fused, adj_t = model_pair(x, edge_index, 1, edge_weight)
    torch.testing.assert_close(
        fused(x, adj_t), reference(x, edge_index, edge_weight=edge_weight),
        rtol=1e-11, atol=1e-12,
    )
