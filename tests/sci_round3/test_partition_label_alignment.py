from test_graph_aware_partition_full_core_coverage import _graph
from gog_fraud.evaluation.graph_aware_partition import GraphAwareHaloPartitioner


def test_local_features_and_labels_follow_global_ids():
    data = _graph()
    plan = GraphAwareHaloPartitioner(data, core_size=2, halo_hops=1, backend="balanced_bfs", stored_bidirectional=True)
    for part in plan:
        assert part.data.y.tolist() == data.y[part.local_to_global].tolist()
        assert part.data.x.tolist() == data.x[part.local_to_global].tolist()
