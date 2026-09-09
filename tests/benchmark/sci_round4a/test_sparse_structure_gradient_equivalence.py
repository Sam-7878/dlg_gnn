import torch

from gog_fraud.models.pygod.exact_reconstruction import exact_dot_product_row_error


def test_sparse_structure_gradient_equivalence(reconstruction_case):
    edge_index, z, adjacency = reconstruction_case
    dense_loss = torch.linalg.vector_norm(adjacency - z @ z.T, dim=1).mean()
    dense_grad, = torch.autograd.grad(dense_loss, z, retain_graph=True)
    sparse_loss = exact_dot_product_row_error(z, edge_index).mean()
    sparse_grad, = torch.autograd.grad(sparse_loss, z)
    torch.testing.assert_close(sparse_grad, dense_grad, rtol=1e-9, atol=1e-10)
