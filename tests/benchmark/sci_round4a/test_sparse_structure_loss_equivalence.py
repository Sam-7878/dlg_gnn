import torch

from gog_fraud.models.pygod.exact_reconstruction import exact_dot_product_row_error


def test_sparse_structure_loss_equivalence(reconstruction_case):
    edge_index, z, adjacency = reconstruction_case
    dense = torch.linalg.vector_norm(adjacency - z @ z.T, dim=1).mean()
    sparse = exact_dot_product_row_error(z, edge_index).mean()
    torch.testing.assert_close(sparse, dense, rtol=1e-10, atol=1e-10)
