#!/usr/bin/env python3
"""
test_exact_sparse_directed_weighted_cases.py

Round M5 Unit Tests: Verifies the exact sparse reconstruction identity across
general graph topologies:
1. Binary directed graph
2. Binary undirected graph
3. Weighted directed graph
4. Graph with self-loops
5. Graph with coalesced parallel edges

Identity:
||A_{i,:} - z_i Z^T||_2^2 = sum_j A_{ij}^2 - 2 * <(A Z)_i, z_i> + z_i (Z^T Z) z_i^T
"""

import pytest
import torch


def compute_dense_row_residual(A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Dense evaluation: ||A_i - z_i Z^T||_2^2 directly via N x N outer product."""
    A_hat = torch.matmul(Z, Z.t())  # [N, N]
    diff = A - A_hat  # [N, N]
    return torch.sum(diff ** 2, dim=1)  # [N]


def compute_exact_sparse_residual(A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """
    Exact sparse evaluation:
    term1 = sum_j A_{ij}^2
    term2 = 2 * <(A Z)_i, z_i>
    term3 = z_i (Z^T Z) z_i^T
    """
    # Term 1: squared row norm
    term1 = torch.sum(A ** 2, dim=1)  # [N]

    # Term 2: sparse-dense cross term
    AZ = torch.matmul(A, Z)  # [N, d]
    term2 = 2.0 * torch.sum(AZ * Z, dim=1)  # [N]

    # Term 3: Gram quadratic form
    G = torch.matmul(Z.t(), Z)  # [d, d]
    ZG = torch.matmul(Z, G)  # [N, d]
    term3 = torch.sum(ZG * Z, dim=1)  # [N]

    return term1 - term2 + term3


@pytest.mark.parametrize("seed", [42, 100, 2026])
@pytest.mark.parametrize("N,d", [(60, 16), (120, 32)])
def test_binary_undirected(seed: int, N: int, d: int):
    torch.manual_seed(seed)
    adj = (torch.rand(N, N) < 0.15).double()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.t()
    Z = torch.randn(N, d, dtype=torch.float64)

    dense = compute_dense_row_residual(adj, Z)
    sparse = compute_exact_sparse_residual(adj, Z)

    max_err = torch.max(torch.abs(dense - sparse)).item()
    assert max_err < 1e-7, f"Binary undirected error too high: {max_err}"


@pytest.mark.parametrize("seed", [42, 100, 2026])
@pytest.mark.parametrize("N,d", [(60, 16), (120, 32)])
def test_binary_directed(seed: int, N: int, d: int):
    torch.manual_seed(seed)
    # Asymmetric directed adjacency matrix
    adj = (torch.rand(N, N) < 0.15).double()
    adj.fill_diagonal_(0.0)
    Z = torch.randn(N, d, dtype=torch.float64)

    dense = compute_dense_row_residual(adj, Z)
    sparse = compute_exact_sparse_residual(adj, Z)

    max_err = torch.max(torch.abs(dense - sparse)).item()
    assert max_err < 1e-7, f"Binary directed error too high: {max_err}"


@pytest.mark.parametrize("seed", [42, 100, 2026])
@pytest.mark.parametrize("N,d", [(60, 16), (120, 32)])
def test_weighted_directed(seed: int, N: int, d: int):
    torch.manual_seed(seed)
    # Weighted directed adjacency: weights in [0.1, 5.0]
    mask = (torch.rand(N, N) < 0.12).double()
    weights = torch.rand(N, N).double() * 4.9 + 0.1
    adj = mask * weights
    adj.fill_diagonal_(0.0)
    Z = torch.randn(N, d, dtype=torch.float64)

    dense = compute_dense_row_residual(adj, Z)
    sparse = compute_exact_sparse_residual(adj, Z)

    max_err = torch.max(torch.abs(dense - sparse)).item()
    assert max_err < 1e-7, f"Weighted directed error too high: {max_err}"


@pytest.mark.parametrize("seed", [42, 100, 2026])
@pytest.mark.parametrize("N,d", [(60, 16), (120, 32)])
def test_graph_with_self_loops(seed: int, N: int, d: int):
    torch.manual_seed(seed)
    adj = (torch.rand(N, N) < 0.15).double()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.t()
    # Add arbitrary self-loops
    diag_weights = torch.rand(N).double() * 2.0
    adj.diagonal().copy_(diag_weights)
    Z = torch.randn(N, d, dtype=torch.float64)

    dense = compute_dense_row_residual(adj, Z)
    sparse = compute_exact_sparse_residual(adj, Z)

    max_err = torch.max(torch.abs(dense - sparse)).item()
    assert max_err < 1e-7, f"Self-loops error too high: {max_err}"


@pytest.mark.parametrize("seed", [42, 100])
def test_coalesced_multigraph(seed: int):
    torch.manual_seed(seed)
    N, d = 50, 16
    # Simulate multigraph where duplicate edges are coalesced by summing weights
    edges_u = torch.randint(0, N, (300,))
    edges_v = torch.randint(0, N, (300,))
    weights = torch.rand(300).double() + 0.5

    adj = torch.zeros(N, N, dtype=torch.float64)
    for u, v, w in zip(edges_u, edges_v, weights):
        adj[u, v] += w

    Z = torch.randn(N, d, dtype=torch.float64)

    dense = compute_dense_row_residual(adj, Z)
    sparse = compute_exact_sparse_residual(adj, Z)

    max_err = torch.max(torch.abs(dense - sparse)).item()
    assert max_err < 1e-7, f"Coalesced multigraph error too high: {max_err}"
