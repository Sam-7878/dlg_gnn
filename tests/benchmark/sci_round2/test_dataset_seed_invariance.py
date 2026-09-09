import torch
from torch_geometric.data import Data
from gog_fraud.experiments.round2_validity import graph_fingerprints


def _graph():
    return Data(x=torch.tensor([[1.], [2.]]), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([0, 1]))


def test_model_seed_does_not_enter_dataset_fingerprint():
    first = graph_fingerprints(_graph(), injection_config={"dataset_seed": 42})
    second = graph_fingerprints(_graph(), injection_config={"dataset_seed": 42})
    assert first == second

