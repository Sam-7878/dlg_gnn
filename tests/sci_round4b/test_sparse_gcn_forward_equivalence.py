import torch

from conftest import model_pair


def test_sparse_gcn_forward_equivalence(message_case):
    x, edge_index = message_case
    reference, fused, adj_t = model_pair(x, edge_index, 2)
    torch.testing.assert_close(
        fused(x, adj_t), reference(x, edge_index), rtol=1e-10, atol=1e-11
    )
