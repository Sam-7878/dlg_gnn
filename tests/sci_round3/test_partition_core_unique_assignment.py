import torch
from test_graph_aware_partition_full_core_coverage import _graph
from gog_fraud.evaluation.graph_aware_partition import GraphAwareHaloPartitioner


def test_assignment_count_is_exactly_one():
    plan = GraphAwareHaloPartitioner(_graph(), core_size=3, halo_hops=2, backend="balanced_bfs", stored_bidirectional=True)
    counts = torch.zeros(6, dtype=torch.int64)
    for part in plan:
        counts[part.core_global_nodes] += 1
    assert counts.tolist() == [1] * 6
