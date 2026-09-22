#!/usr/bin/env python3
"""
test_release_exact_sparse_formula_matches_dense.py

Round M4 Unit Test: Verifies that the canonical exact sparse Gram reconstruction
formula mathematically and numerically matches the dense row residual across
random graphs with varying density and embedding dimensions. Also verifies that
the erroneous Frobenius norm scalar approximation fails.
"""

import json
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
IDENTITY_JSON = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m4" / "release" / "exact_sparse_identity.json"


def compute_dense_row_residual(A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Computes ||A_i - z_i Z^T||_2^2 directly via dense NxN outer product."""
    A_hat = torch.matmul(Z, Z.t())  # [N, N]
    diff = A - A_hat  # [N, N]
    return torch.sum(diff ** 2, dim=1)  # [N]


def compute_exact_sparse_gram_residual(A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """
    Computes ||A_i - z_i Z^T||_2^2 via the exact sparse Gram identity:
    ||A_i||_2^2 - 2 (A Z)_i z_i^T + z_i (Z^T Z) z_i^T
    """
    # 1. Degree / norm of A_i (for binary A, ||A_i||_2^2 = d_i)
    term1 = torch.sum(A ** 2, dim=1)  # [N]

    # 2. Sparse-dense cross term: 2 * (A Z) * Z
    AZ = torch.matmul(A, Z)  # [N, d]
    term2 = 2.0 * torch.sum(AZ * Z, dim=1)  # [N]

    # 3. Gram quadratic form: z_i (Z^T Z) z_i^T
    G = torch.matmul(Z.t(), Z)  # [d, d]
    ZG = torch.matmul(Z, G)  # [N, d]
    term3 = torch.sum(ZG * Z, dim=1)  # [N]

    return term1 - term2 + term3


def compute_erroneous_frobenius_residual(A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Computes the erroneous formula: d_i - 2 (A Z)_i z_i^T + ||Z||_F^2 ||z_i||_2^2."""
    term1 = torch.sum(A ** 2, dim=1)
    AZ = torch.matmul(A, Z)
    term2 = 2.0 * torch.sum(AZ * Z, dim=1)
    fro_sq = torch.sum(Z ** 2)  # ||Z||_F^2
    zi_norm_sq = torch.sum(Z ** 2, dim=1)  # ||z_i||_2^2
    term3_erroneous = fro_sq * zi_norm_sq
    return term1 - term2 + term3_erroneous


@pytest.mark.parametrize("seed", [42, 123, 999])
@pytest.mark.parametrize("N,d", [(50, 8), (100, 16), (200, 32)])
def test_exact_sparse_matches_dense_numerical(seed: int, N: int, d: int):
    torch.manual_seed(seed)

    # Random binary adjacency (undirected, zero diagonal)
    prob = 0.15
    adj = (torch.rand(N, N) < prob).float()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.t()

    # Test with double precision to eliminate single-precision floating point accumulation
    adj = adj.double()
    Z = torch.randn(N, d, dtype=torch.float64)

    dense_res = compute_dense_row_residual(adj, Z)
    sparse_res = compute_exact_sparse_gram_residual(adj, Z)

    # Max absolute error must be negligible in double precision (< 1e-7)
    max_err = torch.max(torch.abs(dense_res - sparse_res)).item()
    rel_err = (torch.norm(dense_res - sparse_res) / torch.norm(dense_res)).item()

    assert max_err < 1e-7, f"Max absolute error too high: {max_err}"
    assert rel_err < 1e-7, f"Relative error too high: {rel_err}"


@pytest.mark.parametrize("seed", [42, 123])
def test_erroneous_formula_strictly_differs(seed: int):
    torch.manual_seed(seed)
    N, d = 50, 8
    adj = (torch.rand(N, N) < 0.2).float()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.t()
    Z = torch.randn(N, d)

    dense_res = compute_dense_row_residual(adj, Z)
    erroneous_res = compute_erroneous_frobenius_residual(adj, Z)

    # The erroneous formula MUST differ significantly from dense truth
    diff = torch.norm(dense_res - erroneous_res).item()
    assert diff > 10.0, f"Erroneous formula unexpectedly close to dense: diff={diff}"


def test_canonical_identity_json_validity():
    assert IDENTITY_JSON.exists(), f"Missing identity JSON: {IDENTITY_JSON}"
    data = json.loads(IDENTITY_JSON.read_text(encoding="utf-8"))
    assert "exact_sparse_identity" in data
    assert "Z^T Z" in data["exact_sparse_identity"]
    assert "||Z||_F^2" not in data["exact_sparse_identity"]
    assert "O(E * d + N * d^2)" in data["complexity"]["arithmetic"]
