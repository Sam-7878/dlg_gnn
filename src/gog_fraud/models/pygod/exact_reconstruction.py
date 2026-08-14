"""Exact adjacency-reconstruction losses without an :math:`N\times N` target.

The functions in this module preserve PyGOD's dense row objective.  They do
not use negative sampling and they do not discard zero entries.  Two exact
execution strategies are provided:

``exact_sparse``
    Closed-form Gram computation for a linear dot-product decoder.

``chunked_exact``
    Row-block materialisation for nonlinear (for example sigmoid) decoders.
    Only ``rows x N`` predictions exist at once; the graph and model remain
    shared and full-graph.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import torch

BackendName = Literal["dense_reference", "exact_sparse", "chunked_exact"]


@dataclass(frozen=True)
class ReconstructionBackend:
    name: BackendName = "exact_sparse"
    score_chunk_size: int = 8192

    @property
    def metadata(self) -> dict[str, object]:
        return {
            **asdict(self),
            "reconstruction_backend": self.name,
            "dense_materialized": self.name == "dense_reference",
            "training_full_graph": True,
            "shared_model": True,
            "approximation_used": False,
        }


BACKENDS = {
    name: ReconstructionBackend(name=name)
    for name in ("dense_reference", "exact_sparse", "chunked_exact")
}


def resolve_backend(name: BackendName, *, score_chunk_size: int = 8192) -> ReconstructionBackend:
    if name not in BACKENDS:
        raise ValueError(f"unknown reconstruction backend: {name}")
    if score_chunk_size <= 0:
        raise ValueError("score_chunk_size must be positive")
    return ReconstructionBackend(name=name, score_chunk_size=int(score_chunk_size))


def _stable_sqrt(squared: torch.Tensor) -> torch.Tensor:
    """Match sqrt for positive inputs and define value/gradient as zero at 0."""
    tiny = torch.finfo(squared.dtype).tiny
    return torch.where(
        squared > 0,
        torch.sqrt(torch.clamp_min(squared, tiny)),
        torch.zeros_like(squared),
    )


def coalesced_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return coalesced source, destination and values as ``to_dense_adj`` does."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.shape[1], dtype=torch.get_default_dtype(), device=edge_index.device
        )
    else:
        edge_weight = edge_weight.to(device=edge_index.device)
    sparse = torch.sparse_coo_tensor(
        edge_index.long(), edge_weight, (num_nodes, num_nodes),
        device=edge_index.device,
    ).coalesce()
    src, dst = sparse.indices()
    return src, dst, sparse.values()


def exact_dot_product_row_squared_error(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    rows: torch.Tensor | None = None,
    edge_weight: torch.Tensor | None = None,
    positive_weight: float = 0.5,
) -> torch.Tensor:
    """Exact dense row squared error for ``A - Z @ Z.T``.

    ``positive_weight=0.5`` reproduces PyGOD's unweighted special case.
    For any other value, positive and zero target entries receive weights
    ``p`` and ``1-p`` respectively, exactly as ``double_recon_loss``.
    Duplicate edges are summed before the positive-entry mask is applied.
    """
    if z.ndim != 2:
        raise ValueError("z must have shape [N, H]")
    n = z.shape[0]
    if rows is None:
        rows = torch.arange(n, device=z.device)
    else:
        rows = rows.to(device=z.device, dtype=torch.long).reshape(-1)
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= n):
        raise IndexError("row index outside embedding range")
    src, dst, values = coalesced_adjacency(edge_index.to(z.device), n, edge_weight)
    values = values.to(dtype=z.dtype, device=z.device)

    inverse = torch.full((n,), -1, dtype=torch.long, device=z.device)
    inverse[rows] = torch.arange(rows.numel(), device=z.device)
    local_src = inverse[src]
    keep = local_src >= 0
    local_src, dst, values = local_src[keep], dst[keep], values[keep]

    z_rows = z[rows]
    gram_term = torch.einsum("bi,ij,bj->b", z_rows, z.T @ z, z_rows)
    edge_dot = (z_rows[local_src] * z[dst]).sum(dim=1)

    if positive_weight == 0.5:
        edge_correction = values.square() - 2.0 * values * edge_dot
        squared = gram_term
    else:
        if not 0.0 <= positive_weight <= 1.0:
            raise ValueError("positive_weight must lie in [0, 1]")
        p = float(positive_weight)
        q = 1.0 - p
        edge_correction = p * values.square() - 2.0 * p * values * edge_dot
        edge_correction = edge_correction + (p - q) * edge_dot.square()
        squared = q * gram_term

    correction_by_row = torch.zeros(rows.numel(), dtype=z.dtype, device=z.device)
    correction_by_row.scatter_add_(0, local_src, edge_correction)
    # Cancellation can create tiny negative values although the exact quantity
    # is non-negative.  Clamping preserves the mathematical objective.
    return torch.clamp_min(squared + correction_by_row, 0.0)


def exact_dot_product_row_error(*args, **kwargs) -> torch.Tensor:
    return _stable_sqrt(exact_dot_product_row_squared_error(*args, **kwargs))


def _row_sparse_correction(
    prediction: torch.Tensor,
    rows: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    values: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    n = prediction.shape[1]
    inverse = torch.full((n,), -1, dtype=torch.long, device=prediction.device)
    inverse[rows] = torch.arange(rows.numel(), device=prediction.device)
    local_src = inverse[src]
    keep = local_src >= 0
    local_src, dst, values = local_src[keep], dst[keep], values[keep]
    pred_edge = prediction[local_src, dst]
    if positive_weight == 0.5:
        base = prediction.square().sum(dim=1)
        correction = (values - pred_edge).square() - pred_edge.square()
    else:
        p, q = float(positive_weight), 1.0 - float(positive_weight)
        base = q * prediction.square().sum(dim=1)
        correction = p * (values - pred_edge).square() - q * pred_edge.square()
    result = base.clone()
    result.scatter_add_(0, local_src, correction)
    return torch.clamp_min(result, 0.0)


def chunked_exact_row_error(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    rows: torch.Tensor | None = None,
    edge_weight: torch.Tensor | None = None,
    positive_weight: float = 0.5,
    sigmoid: bool = False,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Exact row errors with bounded ``chunk_size x N`` materialisation."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    n = z.shape[0]
    if rows is None:
        rows = torch.arange(n, device=z.device)
    else:
        rows = rows.to(device=z.device, dtype=torch.long).reshape(-1)
    src, dst, values = coalesced_adjacency(edge_index.to(z.device), n, edge_weight)
    values = values.to(dtype=z.dtype, device=z.device)
    outputs: list[torch.Tensor] = []
    for start in range(0, rows.numel(), chunk_size):
        chunk = rows[start:start + chunk_size]
        prediction = z[chunk] @ z.T
        if sigmoid:
            prediction = torch.sigmoid(prediction)
        squared = _row_sparse_correction(
            prediction, chunk, src, dst, values, positive_weight
        )
        outputs.append(_stable_sqrt(squared))
    return torch.cat(outputs) if outputs else z.new_empty((0,))


def exact_attribute_error(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    *,
    positive_weight: float = 0.5,
) -> torch.Tensor:
    diff = (x - x_hat).square()
    if positive_weight != 0.5:
        p = float(positive_weight)
        diff = torch.where(x > 0, p * diff, (1.0 - p) * diff)
    return _stable_sqrt(diff.sum(dim=1))


def exact_double_reconstruction_score(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    z_structure: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    weight: float = 0.5,
    positive_weight_attribute: float = 0.5,
    positive_weight_structure: float = 0.5,
    sigmoid_structure: bool = False,
    rows: torch.Tensor | None = None,
    backend: BackendName = "exact_sparse",
    chunk_size: int = 8192,
) -> torch.Tensor:
    if rows is None:
        rows = torch.arange(z_structure.shape[0], device=z_structure.device)
    else:
        rows = rows.to(z_structure.device, dtype=torch.long).reshape(-1)
    attr = exact_attribute_error(
        x[rows], x_hat[rows], positive_weight=positive_weight_attribute
    )
    if backend == "exact_sparse" and not sigmoid_structure:
        struct = exact_dot_product_row_error(
            z_structure, edge_index, rows=rows,
            positive_weight=positive_weight_structure,
        )
    elif backend in ("exact_sparse", "chunked_exact"):
        struct = chunked_exact_row_error(
            z_structure, edge_index, rows=rows,
            positive_weight=positive_weight_structure,
            sigmoid=sigmoid_structure, chunk_size=chunk_size,
        )
    else:
        raise ValueError("dense_reference is intentionally handled by the caller")
    return float(weight) * attr + (1.0 - float(weight)) * struct


def iter_row_chunks(num_nodes: int, chunk_size: int, device: torch.device) -> Iterable[torch.Tensor]:
    for start in range(0, num_nodes, chunk_size):
        yield torch.arange(start, min(start + chunk_size, num_nodes), device=device)
