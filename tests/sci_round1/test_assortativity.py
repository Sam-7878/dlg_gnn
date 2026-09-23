import numpy as np

from gog_fraud.evaluation.fraud_topology import compute_fraud_topology_metrics


def test_perfect_within_class_edges_are_assortative():
    result = compute_fraud_topology_metrics(
        np.array([[0, 1, 2, 3], [1, 0, 3, 2]]), np.array([0, 0, 1, 1])
    )
    assert result.label_assortativity == 1.0

