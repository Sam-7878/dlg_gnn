import torch
from torch_geometric.data import Data

from gog_fraud.models.pygod.shared_reconstruction import SharedCONAD

def test_conad_sparse_full_graph_regression():
    torch.manual_seed(7)
    data = Data(
        x=torch.randn(12, 5),
        edge_index=torch.tensor([[0,1,1,2,3,4,5,6,7,8,9,10,11], [1,0,2,1,4,3,6,5,8,7,10,11,10]]),
        num_nodes=12,
    )
    detector = SharedCONAD(
        epoch=1, hid_dim=4, gpu=-1, score_chunk_size=4,
        gradient_checkpointing=False, message_backend="sparse_fused", r=.2, m=2, k=3,
    )
    detector.fit(data)
    score = detector.decision_function(data)
    assert score.shape == (12,)
    assert torch.isfinite(score).all()
    assert not hasattr(data, "s")
