import torch
from torch_geometric.data import Data

from gog_fraud.models.pygod.shared_reconstruction import (
    SharedAnomalyDAE,
    SharedCONAD,
    SharedDOMINANT,
)


def test_shared_detector_never_materializes_dense_adjacency():
    torch.manual_seed(9)
    data = Data(
        x=torch.randn(8, 3),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 3, 4, 5, 6, 7], [1, 0, 2, 1, 4, 3, 6, 7, 6]]
        ),
        num_nodes=8,
    )
    detector = SharedDOMINANT(
        epoch=2, hid_dim=4, gpu=-1, score_chunk_size=3, verbose=0
    )
    detector.fit(data)
    scores = detector.decision_function(data)
    assert scores.shape == (8,)
    assert torch.isfinite(scores).all()
    assert not hasattr(data, "s")
    metadata = detector.backend_metadata()
    assert metadata["training_full_graph"] is True
    assert metadata["shared_model"] is True
    assert metadata["approximation_used"] is False


def test_nonlinear_and_conad_shared_paths_are_finite_without_dense_target():
    torch.manual_seed(19)
    base = Data(
        x=torch.randn(12, 5),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
             [1, 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 11, 10]]
        ),
        num_nodes=12,
    )
    detectors = (
        SharedAnomalyDAE(epoch=1, emb_dim=4, hid_dim=4, gpu=-1, score_chunk_size=4),
        SharedCONAD(epoch=1, hid_dim=4, gpu=-1, score_chunk_size=4,
                    gradient_checkpointing=False, r=.2, m=2, k=3),
    )
    for detector in detectors:
        data = base.clone()
        detector.fit(data)
        score = detector.decision_function(data)
        assert score.shape == (12,)
        assert torch.isfinite(score).all()
        assert not hasattr(data, "s")
