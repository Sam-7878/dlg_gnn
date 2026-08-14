import numpy as np

from gog_fraud.evaluation.fraud_topology import compute_fraud_topology_metrics


def test_random_mixing_adjusted_homophily_near_zero():
    rng = np.random.default_rng(42)
    labels = np.array([0] * 800 + [1] * 200)
    edges = np.stack([rng.integers(0, 1000, 100_000), rng.integers(0, 1000, 100_000)])
    result = compute_fraud_topology_metrics(edges, labels)
    assert abs(result.adjusted_homophily) < 0.02

