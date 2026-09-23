from test_graph_aware_partition_full_core_coverage import _graph
from gog_fraud.evaluation.graph_aware_partition import GraphAwareHaloPartitioner


def test_each_core_incident_edge_is_present_in_context():
    plan = GraphAwareHaloPartitioner(_graph(), core_size=2, halo_hops=1, backend="balanced_bfs", stored_bidirectional=True)
    for part in plan:
        assert part.stats.covered_core_incident_edges == part.stats.original_core_incident_edges
        assert part.stats.dense_adjacency_elements == part.data.num_nodes ** 2
