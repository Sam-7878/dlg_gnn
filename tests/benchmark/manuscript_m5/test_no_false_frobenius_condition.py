#!/usr/bin/env python3
"""
test_no_false_frobenius_condition.py

Round M5 Unit Test:
1. Verifies that documentation does not contain the mathematically erroneous
   claim that Z^T Z = (||Z||_F^2 / d) I would make z_i (Z^T Z) z_i^T equal ||Z||_F^2 ||z_i||^2.
2. Numerically verifies that if Z^T Z = (||Z||_F^2 / d) I, the quadratic form evaluates
   to (||Z||_F^2 / d) * ||z_i||_2^2, which differs by a factor of 1/d from the erroneous formula.
"""

from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_documentation_lacks_false_frobenius_condition():
    math_doc = REPO_ROOT / "docs/math/exact_sparse_reconstruction.md"
    assert math_doc.exists()
    content = math_doc.read_text(encoding="utf-8")

    # Banned erroneous phrase
    banned = r"Z^T Z = \frac{\|Z\|_F^2}{d} I"
    assert "Z^\\top Z = \\frac{\\|Z\\|_F^2}{d} I" not in content
    assert "Z^T Z = \\frac{||Z||_F^2}{d} I" not in content
    assert "pathological condition" not in content

    # Must contain correct statement
    assert "scalar Frobenius substitution is not valid in general" in content


def test_frobenius_scaling_factor_numerical():
    # Construct an orthogonal matrix where Z^T Z = c * I
    N, d = 100, 16
    torch.manual_seed(42)

    # QR decomposition of random matrix gives orthogonal columns
    Q, _ = torch.linalg.qr(torch.randn(N, d, dtype=torch.float64))
    # Scale Q so rows have non-trivial norms
    scale = 3.5
    Z = Q * scale  # Z^T Z = scale^2 * I_d

    G = torch.matmul(Z.t(), Z)  # scale^2 * I_d
    fro_sq = torch.sum(Z ** 2).item()
    c = fro_sq / d

    # For any row i:
    zi = Z[0:1, :]  # [1, d]
    quad_form = (zi @ G @ zi.t()).item()
    zi_norm_sq = torch.sum(zi ** 2).item()

    # Exact quadratic form is c * ||z_i||^2 = (||Z||_F^2 / d) * ||z_i||^2
    expected_quad = c * zi_norm_sq
    erroneous_scalar = fro_sq * zi_norm_sq

    assert abs(quad_form - expected_quad) < 1e-8
    # With d = 16, erroneous_scalar is 16 times larger!
    ratio = erroneous_scalar / quad_form
    assert abs(ratio - d) < 1e-6, f"Expected ratio {d}, got {ratio}"
