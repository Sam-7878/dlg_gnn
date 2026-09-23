import numpy as np
from test_graph_aware_partition_full_core_coverage import _graph
from gog_fraud.evaluation.graph_aware_partition import GraphAwareHaloPartitioner, reassemble_core_scores


def test_global_node_scores_roundtrip_through_overlapping_halos():
    plan = GraphAwareHaloPartitioner(_graph(), core_size=2, halo_hops=1, backend="balanced_bfs", stored_bidirectional=True)
    parts = list(plan)
    local_scores = [part.local_to_global.numpy().astype(float) + 0.25 for part in parts]
    result = reassemble_core_scores(iter(parts), iter(local_scores), 6)
    np.testing.assert_allclose(result, np.arange(6) + 0.25)
