from test_graph_aware_partition_full_core_coverage import _graph
from gog_fraud.evaluation.graph_aware_partition import GraphAwareHaloPartitioner


def test_one_hop_halo_preserves_all_core_neighbors():
    plan = GraphAwareHaloPartitioner(_graph(), core_size=2, halo_hops=1, backend="balanced_bfs", stored_bidirectional=True)
    for part in plan:
        assert part.stats.core_neighbor_coverage == 1.0
        assert part.stats.core_edge_coverage == 1.0
