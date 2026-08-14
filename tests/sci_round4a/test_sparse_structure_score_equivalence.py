import torch

from gog_fraud.models.pygod.exact_reconstruction import (
    chunked_exact_row_error,
    exact_dot_product_row_error,
)


def test_sparse_structure_score_equivalence(reconstruction_case):
    edge_index, z, adjacency = reconstruction_case
    dense = torch.linalg.vector_norm(adjacency - z @ z.T, dim=1)
    sparse = exact_dot_product_row_error(z, edge_index)
    torch.testing.assert_close(sparse, dense, rtol=1e-10, atol=1e-10)


def test_chunk_size_does_not_change_scores(reconstruction_case):
    edge_index, z, _ = reconstruction_case
    references = [
        chunked_exact_row_error(z, edge_index, chunk_size=size)
        for size in (1, 2, 5)
    ]
    for score in references[1:]:
        torch.testing.assert_close(score, references[0], rtol=1e-12, atol=1e-12)
