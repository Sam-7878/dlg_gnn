import torch

from conftest import model_pair


def test_sparse_gcn_weighted_edge_equivalence(message_case):
    x, edge_index = message_case
    edge_weight = torch.linspace(.2, 2.1, edge_index.shape[1], dtype=x.dtype)
    reference, fused, adj_t = model_pair(x, edge_index, 2, edge_weight)
    torch.testing.assert_close(
        fused(x, adj_t), reference(x, edge_index, edge_weight=edge_weight),
        rtol=1e-10, atol=1e-11,
    )
