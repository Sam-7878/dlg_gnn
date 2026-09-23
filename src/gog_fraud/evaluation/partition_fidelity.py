"""Topology-preservation audit for contiguous induced-subgraph partitions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from gog_fraud.evaluation.fraud_topology import compute_fraud_topology_metrics


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"): value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class PartitionFidelity:
    partition_size: int
    num_partitions: int
    original_num_nodes: int
    original_num_edges: int
    retained_num_edges: int
    cross_partition_edges: int
    edge_retention: float
    cross_partition_ratio: float
    original_non_self_edges: int
    retained_non_self_edges: int
    non_self_edge_retention: float
    connected_components_before: int
    connected_components_after: int
    avg_degree_before: float
    avg_degree_after: float
    positive_ratio_before: float
    positive_ratio_after: float
    edge_homophily_before: float
    edge_homophily_after: float
    fraud_homophily_before: float
    fraud_homophily_after: float
    adjusted_homophily_before: float
    adjusted_homophily_after: float

    def to_dict(self) -> dict[str, Any]: return asdict(self)


def contiguous_partition_ids(num_nodes: int, partition_size: int) -> np.ndarray:
    if num_nodes < 0 or partition_size <= 0: raise ValueError("valid node count and positive partition size are required")
    return np.arange(num_nodes, dtype=np.int64) // int(partition_size)


def retained_edge_mask(edge_index: Any, partition_ids: Any) -> np.ndarray:
    edges, ids = _numpy(edge_index).astype(np.int64), _numpy(partition_ids).astype(np.int64).reshape(-1)
    if edges.ndim != 2 or edges.shape[0] != 2: raise ValueError("edge_index must be [2,E]")
    return ids[edges[0]] == ids[edges[1]]


def _components(num_nodes: int, edge_index: np.ndarray) -> int:
    if num_nodes == 0: return 0
    graph = coo_matrix((np.ones(edge_index.shape[1], dtype=np.uint8), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)).tocsr()
    count, _ = connected_components(graph, directed=False, return_labels=True)
    return int(count)


def audit_contiguous_partition(edge_index: Any, labels: Any, partition_size: int,
                               *, directed: bool = True) -> PartitionFidelity:
    edges = _numpy(edge_index).astype(np.int64); y = _numpy(labels).astype(np.int64).reshape(-1)
    ids = contiguous_partition_ids(len(y), partition_size); keep = retained_edge_mask(edges, ids)
    retained = edges[:, keep]
    before = compute_fraud_topology_metrics(edges, y, directed=directed)
    after = compute_fraud_topology_metrics(retained, y, directed=directed)
    non_self = edges[0] != edges[1]; retained_non_self = non_self & keep
    edge_count, retained_count = edges.shape[1], retained.shape[1]
    return PartitionFidelity(
        partition_size=int(partition_size), num_partitions=int(ids.max() + 1 if len(ids) else 0),
        original_num_nodes=len(y), original_num_edges=edge_count, retained_num_edges=retained_count,
        cross_partition_edges=int(edge_count - retained_count),
        edge_retention=float(retained_count / edge_count) if edge_count else float("nan"),
        cross_partition_ratio=float((edge_count - retained_count) / edge_count) if edge_count else float("nan"),
        original_non_self_edges=int(non_self.sum()), retained_non_self_edges=int(retained_non_self.sum()),
        non_self_edge_retention=float(retained_non_self.sum() / non_self.sum()) if non_self.sum() else float("nan"),
        connected_components_before=_components(len(y), edges), connected_components_after=_components(len(y), retained),
        avg_degree_before=before.avg_degree, avg_degree_after=after.avg_degree,
        positive_ratio_before=before.positive_ratio, positive_ratio_after=after.positive_ratio,
        edge_homophily_before=before.edge_homophily, edge_homophily_after=after.edge_homophily,
        fraud_homophily_before=before.fraud_homophily, fraud_homophily_after=after.fraud_homophily,
        adjusted_homophily_before=before.adjusted_homophily, adjusted_homophily_after=after.adjusted_homophily,
    )
