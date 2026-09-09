import torch
from torch_geometric.data import Data
from gog_fraud.evaluation.graph_aware_partition import GraphAwareHaloPartitioner


def _graph():
    edge = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    return Data(x=torch.arange(12, dtype=torch.float32).view(6, 2), y=torch.tensor([0, 1, 0, 1, 0, 1]), edge_index=edge)


def test_every_node_is_in_one_core():
    plan = GraphAwareHaloPartitioner(_graph(), core_size=2, halo_hops=1, backend="balanced_bfs", stored_bidirectional=True)
    plan.assert_unique_core_assignment()
    cores = torch.cat([part.core_global_nodes for part in plan])
    assert torch.equal(torch.sort(cores).values, torch.arange(6))
