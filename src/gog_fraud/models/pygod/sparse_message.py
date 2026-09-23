"""Exact GCN message backends for SCI Round 4B.

``sparse_fused`` uses the *same* PyG COO normalization routine as the
historical GCNConv and converts the normalized operator to ``SparseTensor``.
All GCN layers then share that operator and execute fused sparse matrix
multiplication, avoiding an explicit ``[E, H]`` message tensor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.nn import GCN
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor

MessageBackendName = Literal["pyg_coo_reference", "sparse_fused"]


@dataclass(frozen=True)
class MessageBackend:
    name: MessageBackendName = "sparse_fused"
    normalized_adjacency_cached: bool = True

    @property
    def metadata(self) -> dict[str, object]:
        return {
            **asdict(self),
            "message_backend": self.name,
            "edge_expanded_messages": self.name == "pyg_coo_reference",
            "full_graph": True,
            "approximation_used": False,
        }


def resolve_message_backend(name: MessageBackendName) -> MessageBackend:
    if name not in ("pyg_coo_reference", "sparse_fused"):
        raise ValueError(f"unknown message backend: {name}")
    return MessageBackend(name=name, normalized_adjacency_cached=name == "sparse_fused")


def normalized_sparse_adjt(
    edge_index: Tensor,
    num_nodes: int,
    *,
    edge_weight: Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    improved: bool = False,
    add_self_loops: bool = True,
) -> SparseTensor:
    """Build exact PyG-normalized transposed adjacency for fused SpMM.

    Calling PyG's tensor ``gcn_norm`` first preserves its exact treatment of
    remaining self-loops, directed degree, duplicate edges, and edge weights.
    Coalescing afterwards is valid because normalized GCN aggregation is
    linear in duplicate edge weights.
    """
    source_device = edge_index.device
    if edge_weight is not None:
        edge_weight = edge_weight.to(source_device)
    normalized_index, normalized_weight = gcn_norm(
        edge_index,
        edge_weight,
        num_nodes=num_nodes,
        improved=improved,
        add_self_loops=add_self_loops,
        flow="source_to_target",
        dtype=dtype,
    )
    assert normalized_weight is not None
    # SparseTensor follows PyG's adj_t contract: row=target, col=source.
    adj_t = SparseTensor(
        row=normalized_index[1],
        col=normalized_index[0],
        value=normalized_weight,
        sparse_sizes=(num_nodes, num_nodes),
        is_sorted=False,
    ).coalesce()
    if device is not None:
        adj_t = adj_t.to(device)
    return adj_t


class SparseFusedGCN(GCN):
    """PyG GCN whose layers consume a pre-normalized shared SparseTensor."""

    def __init__(self, *args, **kwargs):
        # Normalization/self-loops are performed once by normalized_sparse_adjt.
        kwargs["normalize"] = False
        kwargs["add_self_loops"] = False
        kwargs["cached"] = False
        super().__init__(*args, **kwargs)


class AutoSparseFusedGCN(SparseFusedGCN):
    """Drop-in PyG ``GCN`` backbone accepting either COO or SparseTensor.

    Historical PyGOD detectors call their backbone with a tensor edge index.
    This adapter preserves that public contract while converting each sampled
    (or full-batch) graph with PyG's reference normalization before fused SpMM.
    It does not change detector sampling, loss, score, or decoder semantics.
    """

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, Tensor):
            edge_index = normalized_sparse_adjt(
                edge_index, x.size(0), edge_weight=edge_weight,
                dtype=x.dtype, device=x.device,
            )
        return super().forward(x, edge_index)


class MessageGraphCache:
    """One normalized operator per static graph/device/dtype."""

    def __init__(self):
        self._key: tuple | None = None
        self._adj_t: SparseTensor | None = None

    def clear(self) -> None:
        self._key = None
        self._adj_t = None

    def get(
        self,
        edge_index: Tensor,
        num_nodes: int,
        *,
        edge_weight: Tensor | None,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> SparseTensor:
        key = (
            int(edge_index.data_ptr()), tuple(edge_index.shape), int(num_nodes),
            str(dtype), str(device),
            None if edge_weight is None else int(edge_weight.data_ptr()),
        )
        if self._key != key or self._adj_t is None:
            self._adj_t = normalized_sparse_adjt(
                edge_index, num_nodes, edge_weight=edge_weight,
                dtype=dtype, device=device,
            )
            self._key = key
        return self._adj_t


def estimate_coo_message_bytes(num_edges: int, hidden_dim: int, *, bytes_per_value: int = 4) -> int:
    return int(num_edges) * int(hidden_dim) * int(bytes_per_value)


def estimate_sparse_operator_bytes(num_nodes: int, num_edges_with_loops: int, *, bytes_per_value: int = 4) -> int:
    # CSR rowptr int64 + col int64 + values.
    return (int(num_nodes) + 1) * 8 + int(num_edges_with_loops) * (8 + int(bytes_per_value))
