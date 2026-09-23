import numpy as np
import torch
from gog_fraud.data.dgraphfin_aligned import load_dgraphfin_aligned


def test_background_filter_preserves_edge_time_and_official_splits(tmp_path):
    path = tmp_path / "dgraphfin.npz"
    np.savez(path, x=np.arange(15, dtype=np.float32).reshape(5, 3), y=np.array([0, 2, 1, 3, 0]),
             edge_index=np.array([[0, 2], [2, 4], [0, 1], [3, 4]]),
             edge_type=np.array([7, 8, 9, 10]), edge_timestamp=np.array([11, 12, 13, 14]),
             train_mask=np.array([0]), valid_mask=np.array([2]), test_mask=np.array([4]))
    data = load_dgraphfin_aligned(path)
    assert data.original_node_id.tolist() == [0, 2, 4]
    assert data.edge_index.tolist() == [[0, 1], [1, 2]]
    assert data.edge_timestamp.tolist() == [11, 12]
    assert data.edge_type.tolist() == [7, 8]
    assert data.train_mask.tolist() == [True, False, False]
    assert data.val_mask.tolist() == [False, True, False]
    assert data.test_mask.tolist() == [False, False, True]
    assert not torch.any(data.train_mask & data.val_mask)
