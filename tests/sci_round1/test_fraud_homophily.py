import numpy as np

from gog_fraud.evaluation.fraud_topology import compute_fraud_topology_metrics


def test_imbalanced_graph_raw_high_but_fraud_homophily_low():
    labels = np.array([0] * 98 + [1] * 2)
    normal_src = np.arange(96)
    normal_dst = np.arange(1, 97)
    fraud_src = np.array([98, 99])
    fraud_dst = np.array([0, 1])
    edges = np.stack([np.concatenate([normal_src, fraud_src]), np.concatenate([normal_dst, fraud_dst])])
    result = compute_fraud_topology_metrics(edges, labels, directed=True)
    assert result.edge_homophily > 0.95
    assert result.fraud_homophily == 0.0
    assert result.mix_fraud_to_normal == 1.0


def test_clustered_fraud_has_high_fraud_homophily():
    labels = np.array([0, 0, 0, 1, 1, 1])
    edges = np.array([[0, 1, 3, 4, 5], [1, 2, 4, 5, 3]])
    result = compute_fraud_topology_metrics(edges, labels)
    assert result.fraud_homophily == 1.0
    assert result.mix_fraud_to_normal == 0.0

