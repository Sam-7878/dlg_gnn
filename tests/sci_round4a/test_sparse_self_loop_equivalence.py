import torch
from torch_geometric.utils import to_dense_adj

from gog_fraud.models.pygod.exact_reconstruction import exact_dot_product_row_error


def test_self_loops_and_duplicate_edges_match_to_dense_adj():
    edge_index = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]])
    z = torch.tensor([[0.3, -0.2], [0.7, 0.4]], dtype=torch.float64)
    adjacency = to_dense_adj(edge_index, max_num_nodes=2)[0].to(z)
    dense = torch.linalg.vector_norm(adjacency - z @ z.T, dim=1)
    sparse = exact_dot_product_row_error(z, edge_index)
    torch.testing.assert_close(sparse, dense, rtol=1e-12, atol=1e-12)
