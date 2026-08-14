import torch

from gog_fraud.models.pygod.exact_reconstruction import exact_dot_product_row_error


def test_sparse_zero_residual_gradient_is_finite_and_zero():
    z = torch.eye(4, dtype=torch.float64, requires_grad=True)
    nodes = torch.arange(4)
    edge_index = torch.stack([nodes, nodes])
    score = exact_dot_product_row_error(z, edge_index)
    assert torch.equal(score, torch.zeros_like(score))
    score.mean().backward()
    assert torch.isfinite(z.grad).all()
    assert torch.equal(z.grad, torch.zeros_like(z.grad))
