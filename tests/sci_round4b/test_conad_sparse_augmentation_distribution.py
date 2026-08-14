import numpy as np
import torch
from torch_geometric.data import Data

from gog_fraud.models.pygod.shared_reconstruction import SharedCONAD


def test_conad_sparse_augmentation_matches_bernoulli_edge_rate_without_dense_copy():
    n, rate, mean_edges = 200, .8, 10
    detector = SharedCONAD(
        epoch=1, gpu=-1, r=rate, m=mean_edges, k=3,
        gradient_checkpointing=False, message_backend="sparse_fused",
    )
    data = Data(x=torch.randn(n, 4), edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=n)
    observed = []
    for seed in range(80):
        torch.manual_seed(seed)
        _, edge_index, _ = detector._sparse_data_augmentation(data)
        observed.append(edge_index.shape[1])
    # Unconditional expectation: N rows * P(high) * E[Binomial(N,m/N)].
    expected = n * (rate / 4) * mean_edges
    assert abs(float(np.mean(observed)) - expected) / expected < .08
    assert not hasattr(data, "s")
