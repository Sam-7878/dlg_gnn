import torch

from gog_fraud.models.pygod.exact_reconstruction import (
    chunked_exact_row_error,
    exact_dot_product_row_error,
)


def _dense_weighted(adjacency, prediction, positive_weight):
    diff = (adjacency - prediction).square()
    diff = torch.where(
        adjacency > 0,
        positive_weight * diff,
        (1.0 - positive_weight) * diff,
    )
    return torch.sqrt(diff.sum(dim=1))


def test_weighted_linear_reconstruction_equivalence(reconstruction_case):
    edge_index, z, adjacency = reconstruction_case
    dense = _dense_weighted(adjacency, z @ z.T, 0.8)
    sparse = exact_dot_product_row_error(z, edge_index, positive_weight=0.8)
    torch.testing.assert_close(sparse, dense, rtol=1e-10, atol=1e-10)


def test_weighted_sigmoid_chunked_equivalence(reconstruction_case):
    edge_index, z, adjacency = reconstruction_case
    dense = _dense_weighted(adjacency, torch.sigmoid(z @ z.T), 0.8)
    sparse = chunked_exact_row_error(
        z, edge_index, positive_weight=0.8, sigmoid=True, chunk_size=2
    )
    torch.testing.assert_close(sparse, dense, rtol=1e-10, atol=1e-10)
