import numpy as np
from gog_fraud.evaluation.graph_conventions import stored_edge_audit
from gog_fraud.evaluation.fraud_topology import compute_fraud_topology_metrics


def test_bidirectional_storage_is_detected_and_not_mirrored_again():
    edges = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    assert stored_edge_audit(edges, 3)["contains_reverse_edges"]
    direct = compute_fraud_topology_metrics(edges, [0, 0, 1], directed=True)
    assert direct.num_edges == 4

