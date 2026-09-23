"""Memory-safe topology characterization for imbalanced binary fraud graphs."""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class FraudTopologyMetrics:
    num_nodes: int
    num_edges: int
    positive_ratio: float
    avg_degree: float
    edge_homophily: float
    fraud_homophily: float
    normal_homophily: float
    mix_fraud_to_normal: float
    mix_normal_to_fraud: float
    adjusted_homophily: float
    label_assortativity: float
    directed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ratio(numerator: int | float, denominator: int | float, name: str) -> float:
    if denominator == 0:
        warnings.warn(f"{name} is undefined because its conditioning class has no incident edges")
        return float("nan")
    return float(numerator / denominator)


def compute_fraud_topology_metrics(
    edge_index: Any,
    labels: Any,
    *,
    directed: bool = True,
    symmetric_for_undirected: bool = True,
) -> FraudTopologyMetrics:
    """Compute raw, class-conditioned, adjusted, and assortativity metrics.

    Edges are interpreted as source->target observations. For an undirected graph,
    set ``directed=False``. If only one orientation per undirected edge is stored,
    the default mirrors it so the class-conditioned denominators are symmetric.
    Duplicate edges are retained because they represent the graph's observed edge
    multiplicity, matching conventional edge homophily.
    """
    edges = _numpy(edge_index).astype(np.int64, copy=False)
    y = _numpy(labels).astype(np.int64, copy=False).reshape(-1)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if y.size == 0 or not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be a non-empty binary vector")
    if edges.size and (edges.min() < 0 or edges.max() >= y.size):
        raise ValueError("edge_index contains a node outside the labels vector")
    if not directed and symmetric_for_undirected and edges.shape[1]:
        edges = np.concatenate((edges, edges[::-1]), axis=1)

    src, dst = edges
    num_edges = int(src.size)
    positive_ratio = float(y.mean())
    avg_degree = float(num_edges / y.size) if directed else float(num_edges / y.size)
    if num_edges == 0:
        warnings.warn("all edge-based topology metrics are undefined for an empty graph")
        nan = float("nan")
        return FraudTopologyMetrics(y.size, 0, positive_ratio, 0.0, nan, nan, nan, nan, nan, nan, nan, directed)

    src_y, dst_y = y[src], y[dst]
    counts = np.zeros((2, 2), dtype=np.float64)
    np.add.at(counts, (src_y, dst_y), 1.0)
    row = counts.sum(axis=1)
    col = counts.sum(axis=0)
    same = float(counts[0, 0] + counts[1, 1])
    edge_homophily = same / num_edges
    fraud_homophily = _ratio(counts[1, 1], row[1], "fraud_homophily")
    normal_homophily = _ratio(counts[0, 0], row[0], "normal_homophily")
    mix_10 = _ratio(counts[1, 0], row[1], "mix_fraud_to_normal")
    mix_01 = _ratio(counts[0, 1], row[0], "mix_normal_to_fraud")

    # Chance agreement under independent endpoint labels. Endpoint marginals are
    # used instead of global node prevalence so directed degree imbalance is not
    # silently discarded. For symmetric undirected storage they are identical.
    source_p = row / num_edges
    target_p = col / num_edges
    expected = float(np.dot(source_p, target_p))
    adjusted = _ratio(edge_homophily - expected, 1.0 - expected, "adjusted_homophily")

    # Newman's categorical assortativity for the 2x2 directed mixing matrix:
    # r = (Tr(e) - sum_i a_i b_i) / (1 - sum_i a_i b_i).
    assortativity = adjusted
    return FraudTopologyMetrics(
        num_nodes=int(y.size), num_edges=num_edges, positive_ratio=positive_ratio,
        avg_degree=avg_degree, edge_homophily=edge_homophily,
        fraud_homophily=fraud_homophily, normal_homophily=normal_homophily,
        mix_fraud_to_normal=mix_10, mix_normal_to_fraud=mix_01,
        adjusted_homophily=adjusted, label_assortativity=assortativity,
        directed=bool(directed),
    )

