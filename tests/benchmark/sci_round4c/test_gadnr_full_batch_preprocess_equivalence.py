import copy

import torch
from pygod.nn import GADNRBase as UpstreamGADNRBase
from torch_geometric.data import Data

from gog_fraud.models.pygod.gadnr import GADNR


def test_exact_degree_bincount_matches_upstream_full_batch_preprocess():
    data = Data(
        x=torch.tensor(
            [[1.0, 2.0], [0.5, 0.5], [3.0, 1.0], [1.0, 0.0]],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 3], [1, 0, 2, 3, 2]], dtype=torch.long
        ),
    )
    reference, _, reference_degree, _ = UpstreamGADNRBase.process_graph(
        copy.deepcopy(data)
    )

    detector = GADNR(epoch=1, gpu=-1, batch_size=0, verbose=0)
    detector.batch_size = data.num_nodes
    optimized = detector.process_graph(copy.deepcopy(data))

    assert torch.equal(optimized.edge_index, reference.edge_index)
    assert torch.equal(optimized.x, reference.x)
    assert torch.equal(detector.neighbor_num_list.cpu(), reference_degree)
    assert detector.neighbor_dict == {}
    assert detector.id_mapping == {}
    assert detector.full_batch_preprocess_backend_ == "exact_degree_bincount"

