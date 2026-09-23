"""METIS core assignment with exact halo-context extraction for SCI runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops

try:
    from torch_sparse import SparseTensor
except ImportError:  # pragma: no cover - exercised by explicit fallback tests only
    SparseTensor = None


class PartitionBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class CoreHaloStats:
    partition_id: int
    partition_strategy: str
    num_core_nodes: int
    num_halo_nodes: int
    total_local_nodes: int
    halo_ratio: float
    core_edge_coverage: float
    core_neighbor_coverage: float
    original_core_incident_edges: int
    covered_core_incident_edges: int
    dense_adjacency_elements: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CoreHaloSubgraph:
    data: Data
    core_global_nodes: Tensor
    local_to_global: Tensor
    core_local_index: Tensor
    stats: CoreHaloStats


def _validate_edge_index(edge_index: Tensor, num_nodes: int) -> None:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise IndexError("edge_index is outside [0, num_nodes)")


def _symmetric_edges(edge_index: Tensor, *, stored_bidirectional: bool) -> Tensor:
    non_self = edge_index[:, edge_index[0] != edge_index[1]].cpu()
    if stored_bidirectional:
        return non_self
    return torch.cat((non_self, non_self.flip(0)), dim=1)


def _metis_assignment(edge_index: Tensor, num_nodes: int, num_parts: int,
                      *, stored_bidirectional: bool) -> Tensor:
    if SparseTensor is None:
        raise ImportError("graph-aware METIS partition requires torch-sparse")
    symmetric = _symmetric_edges(edge_index, stored_bidirectional=stored_bidirectional)
    adjacency = SparseTensor(row=symmetric[0], col=symmetric[1],
                             sparse_sizes=(num_nodes, num_nodes)).coalesce()
    rowptr, col, _ = adjacency.csr()
    try:
        assignment = torch.ops.torch_sparse.partition(rowptr, col, None, int(num_parts), False)
    except (AttributeError, RuntimeError) as exc:
        raise ImportError("torch-sparse was built without METIS partition support") from exc
    return assignment.to(torch.long).cpu()


def _balanced_bfs_assignment(edge_index: Tensor, num_nodes: int, core_size: int,
                             *, stored_bidirectional: bool) -> Tensor:
    """Deterministic graph-aware fallback; intended for tests/smaller graphs."""
    if SparseTensor is None:
        raise ImportError("balanced BFS fallback requires torch-sparse CSR support")
    symmetric = _symmetric_edges(edge_index, stored_bidirectional=stored_bidirectional)
    adjacency = SparseTensor(row=symmetric[0], col=symmetric[1],
                             sparse_sizes=(num_nodes, num_nodes)).coalesce()
    rowptr, col, _ = adjacency.csr()
    assignment = torch.full((num_nodes,), -1, dtype=torch.long)
    degree = rowptr[1:] - rowptr[:-1]
    remaining = set(range(num_nodes)); part_id = 0
    while remaining:
        seed = max(remaining, key=lambda node: (int(degree[node]), -node))
        queue, queued, chosen = [seed], {seed}, []
        cursor = 0
        while cursor < len(queue) and len(chosen) < core_size:
            node = queue[cursor]; cursor += 1
            if node not in remaining:
                continue
            remaining.remove(node); chosen.append(node)
            start, end = int(rowptr[node]), int(rowptr[node + 1])
            for neighbor in col[start:end].tolist():
                if neighbor in remaining and neighbor not in queued:
                    queue.append(neighbor); queued.add(neighbor)
        while len(chosen) < core_size and remaining:
            node = min(remaining); remaining.remove(node); chosen.append(node)
        assignment[torch.tensor(chosen, dtype=torch.long)] = part_id
        part_id += 1
    return assignment


class GraphAwareHaloPartitioner:
    """Creates non-overlapping METIS cores and overlapping k-hop context."""

    def __init__(self, data: Data, *, core_size: int, halo_hops: int = 1,
                 backend: str = "metis", stored_bidirectional: bool = False,
                 max_expanded_nodes: int | None = None) -> None:
        if core_size <= 0 or halo_hops < 0:
            raise ValueError("core_size must be positive and halo_hops non-negative")
        if data.edge_index is None:
            raise ValueError("data.edge_index is required")
        self.data, self.core_size, self.halo_hops = data, int(core_size), int(halo_hops)
        self.backend, self.strategy = backend, f"graph_aware_halo_{backend}"
        self.stored_bidirectional = bool(stored_bidirectional)
        self.max_expanded_nodes = max_expanded_nodes
        self.num_nodes = int(data.num_nodes)
        edge_index = data.edge_index.cpu().to(torch.long)
        _validate_edge_index(edge_index, self.num_nodes)
        self.original_edge_index = edge_index
        num_parts = max(1, int(np.ceil(self.num_nodes / self.core_size)))
        if backend == "metis":
            self.assignment = _metis_assignment(edge_index, self.num_nodes, num_parts,
                                                stored_bidirectional=self.stored_bidirectional)
        elif backend == "balanced_bfs":
            self.assignment = _balanced_bfs_assignment(edge_index, self.num_nodes, self.core_size,
                                                       stored_bidirectional=self.stored_bidirectional)
        else:
            raise ValueError(f"unsupported graph-aware backend: {backend}")
        if self.assignment.numel() != self.num_nodes or (self.assignment < 0).any():
            raise RuntimeError("partition backend did not assign every node")
        self.partition_ids = torch.unique(self.assignment, sorted=True).tolist()
        symmetric = _symmetric_edges(edge_index, stored_bidirectional=self.stored_bidirectional)
        self.context_adjacency = SparseTensor(row=symmetric[0], col=symmetric[1],
                                              sparse_sizes=(self.num_nodes, self.num_nodes)).coalesce()
        self.original_adjacency = SparseTensor(row=edge_index[0], col=edge_index[1],
                                               sparse_sizes=(self.num_nodes, self.num_nodes))

    def __len__(self) -> int:
        return len(self.partition_ids)

    def core_counts(self) -> Tensor:
        return torch.bincount(self.assignment, minlength=max(self.partition_ids) + 1)

    def assert_unique_core_assignment(self) -> None:
        counts = torch.zeros(self.num_nodes, dtype=torch.int16)
        for part_id in self.partition_ids:
            counts[self.assignment == part_id] += 1
        if not torch.all(counts == 1):
            raise AssertionError("every node must be assigned to exactly one core")

    def _expand(self, core: Tensor) -> Tensor:
        selected = torch.unique(core, sorted=True)
        frontier = selected
        for _ in range(self.halo_hops):
            if frontier.numel() == 0:
                break
            _, neighbors, _ = self.context_adjacency[frontier].coo()
            selected = torch.unique(torch.cat((selected, neighbors)), sorted=True)
            frontier = neighbors
        return selected

    def _nodes_and_stats(self, partition_id: int) -> tuple[Tensor, Tensor, CoreHaloStats]:
        core = torch.nonzero(self.assignment == int(partition_id), as_tuple=False).flatten()
        if core.numel() == 0:
            raise ValueError(f"empty core partition: {partition_id}")
        expanded = self._expand(core)
        halo = expanded[self.assignment[expanded] != int(partition_id)]
        # Core rows first are required by PyGOD/NeighborLoader score semantics.
        local_nodes = torch.cat((core, halo))
        _, original_neighbors, _ = self.context_adjacency[core].coo()
        original_incident = int(original_neighbors.numel())
        if self.halo_hops >= 1:
            covered_neighbors = original_incident
        else:
            covered_neighbors = int((self.assignment[original_neighbors] == int(partition_id)).sum())
        coverage = covered_neighbors / original_incident if original_incident else 1.0
        stats = CoreHaloStats(
            partition_id=int(partition_id), partition_strategy=self.strategy,
            num_core_nodes=int(core.numel()), num_halo_nodes=int(halo.numel()),
            total_local_nodes=int(local_nodes.numel()), halo_ratio=float(halo.numel() / core.numel()),
            core_edge_coverage=float(coverage), core_neighbor_coverage=float(coverage),
            original_core_incident_edges=original_incident,
            covered_core_incident_edges=covered_neighbors,
            dense_adjacency_elements=int(local_nodes.numel() ** 2),
        )
        return core, local_nodes, stats

    def measure(self, partition_id: int) -> CoreHaloStats:
        return self._nodes_and_stats(partition_id)[2]

    def build(self, partition_id: int) -> CoreHaloSubgraph:
        core, local_nodes, stats = self._nodes_and_stats(partition_id)
        if self.max_expanded_nodes is not None and local_nodes.numel() > self.max_expanded_nodes:
            raise PartitionBudgetExceeded(
                f"partition {partition_id}: core+halo={local_nodes.numel()} exceeds explicit budget "
                f"{self.max_expanded_nodes}; choose a new declared core budget")
        sub_adj = self.original_adjacency[local_nodes][:, local_nodes]
        row, col, _ = sub_adj.coo()
        edge_index = torch.stack((row, col), dim=0).to(torch.long)
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=local_nodes.numel())
        local_map = torch.full((self.num_nodes,), -1, dtype=torch.long)
        local_map[local_nodes] = torch.arange(local_nodes.numel())
        core_local = local_map[core]
        if (core_local < 0).any():
            raise AssertionError("core nodes missing from local subgraph")
        local = Data(x=self.data.x[local_nodes].clone(), y=self.data.y[local_nodes].clone(),
                     edge_index=edge_index, num_nodes=int(local_nodes.numel()))
        for name in ("eval_mask", "train_mask", "val_mask", "test_mask", "node_timestamp"):
            value = getattr(self.data, name, None)
            if torch.is_tensor(value) and value.size(0) == self.num_nodes:
                setattr(local, name, value[local_nodes].clone())
        local.global_node_id = local_nodes
        local.core_mask = torch.zeros(local_nodes.numel(), dtype=torch.bool)
        local.core_mask[core_local] = True
        local.partition_id = int(partition_id)

        return CoreHaloSubgraph(local, core, local_nodes, core_local, stats)

    def __iter__(self) -> Iterator[CoreHaloSubgraph]:
        for partition_id in self.partition_ids:
            yield self.build(int(partition_id))


def reassemble_core_scores(partitions: Iterator[CoreHaloSubgraph], local_scores: Iterator[np.ndarray],
                           num_nodes: int) -> np.ndarray:
    output = np.full(num_nodes, np.nan, dtype=float)
    counts = np.zeros(num_nodes, dtype=np.int16)
    for part, score in zip(partitions, local_scores, strict=True):
        values = np.asarray(score, dtype=float).reshape(-1)
        if values.size != part.data.num_nodes:
            raise ValueError("partition score length does not match local node count")
        global_core = part.core_global_nodes.numpy()
        output[global_core] = values[part.core_local_index.numpy()]
        counts[global_core] += 1
    if not np.all(counts == 1):
        raise AssertionError("every original node must contribute exactly one core score")
    if not np.isfinite(output).all():
        raise ValueError("assembled score contains non-finite values")
    return output
