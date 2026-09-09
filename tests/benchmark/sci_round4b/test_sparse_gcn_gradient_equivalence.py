import torch

from conftest import model_pair


def test_sparse_gcn_gradient_equivalence(message_case):
    x, edge_index = message_case
    x_reference = x.clone().requires_grad_(True)
    x_fused = x.clone().requires_grad_(True)
    reference, fused, adj_t = model_pair(x, edge_index, 4)
    reference(x_reference, edge_index).square().mean().backward()
    fused(x_fused, adj_t).square().mean().backward()
    torch.testing.assert_close(x_fused.grad, x_reference.grad, rtol=1e-9, atol=1e-10)
    for expected, actual in zip(reference.parameters(), fused.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-9, atol=1e-10)
