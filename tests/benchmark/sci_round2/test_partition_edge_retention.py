import numpy as np
from gog_fraud.evaluation.partition_fidelity import audit_contiguous_partition


def test_cross_partition_edges_are_counted_exactly():
    edges = np.array([[0, 1, 2, 3, 1], [1, 2, 3, 0, 0]])
    result = audit_contiguous_partition(edges, np.array([0, 0, 1, 1]), 2)
    assert result.retained_num_edges == 3
    assert result.cross_partition_edges == 2
    assert result.edge_retention == 0.6
