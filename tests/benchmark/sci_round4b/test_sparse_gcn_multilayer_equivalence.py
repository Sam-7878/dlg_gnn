import pytest
import torch

from conftest import model_pair


@pytest.mark.parametrize("layers", [1, 2, 4, 6])
def test_sparse_gcn_multilayer_equivalence(message_case, layers):
    x, edge_index = message_case
    reference, fused, adj_t = model_pair(x, edge_index, layers)
    torch.testing.assert_close(fused(x, adj_t), reference(x, edge_index), rtol=1e-9, atol=1e-10)
