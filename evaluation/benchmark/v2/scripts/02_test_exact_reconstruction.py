#!/usr/bin/env python3
"""02_test_exact_reconstruction.py: Exact structural reconstruction equivalence test (Phase C-01).

Compares:
  Dense reference (A - Z @ Z.T)
vs
  Exact sparse / Gram closed-form implementation (without N x N allocation)

Verifies:
  - total reconstruction loss
  - per-node reconstruction score
  - gradients with respect to Z
  - max absolute difference
  - max relative difference
Tolerance: atol = 1e-5, rtol = 1e-4.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

# Ensure src is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gog_fraud.models.pygod.exact_reconstruction import (
    exact_dot_product_row_error,
    exact_dot_product_row_squared_error,
)

OUTPUT_DIR = REPO_ROOT / "evaluation" / "benchmark" / "v2" / "environment" / "journal_cuda"


def test_reconstruction_equivalence(
    num_nodes: int = 120,
    hid_dim: int = 32,
    edge_density: float = 0.05,
    seed: int = 42,
    device: str = "cuda:1",
) -> dict:
    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Create random sparse graph
    adj_prob = torch.rand(num_nodes, num_nodes, device=dev)
    adj_mask = (adj_prob < edge_density).triu(diagonal=1)
    adj_mask = adj_mask | adj_mask.T  # symmetric undirected graph
    edge_index = torch.nonzero(adj_mask).T.contiguous()
    A_dense = adj_mask.to(torch.float64)

    # 1. Squared error equivalence (float64 for numerical precision, float32 for runtime)
    results = {}
    for dtype_name, dtype in [("float64", torch.float64), ("float32", torch.float32)]:
        # --- Dense reference ---
        Z_dense = torch.randn(num_nodes, hid_dim, dtype=dtype, device=dev, requires_grad=True)
        A_d = A_dense.to(dtype=dtype, device=dev)
        A_hat = Z_dense @ Z_dense.T
        dense_row_sq = (A_d - A_hat).square().sum(dim=1)
        dense_loss = dense_row_sq.mean()
        dense_loss.backward()
        grad_dense = Z_dense.grad.clone()

        # --- Exact sparse / Gram reference ---
        Z_sparse = Z_dense.detach().clone().requires_grad_(True)
        sparse_row_sq = exact_dot_product_row_squared_error(
            Z_sparse, edge_index, positive_weight=0.5
        )
        sparse_loss = sparse_row_sq.mean()
        sparse_loss.backward()
        grad_sparse = Z_sparse.grad.clone()

        # Differences
        score_abs_diff = (dense_row_sq - sparse_row_sq).abs().max().item()
        score_rel_diff = ((dense_row_sq - sparse_row_sq).abs() / (dense_row_sq.abs() + 1e-12)).max().item()
        loss_abs_diff = abs(dense_loss.item() - sparse_loss.item())
        grad_abs_diff = (grad_dense - grad_sparse).abs().max().item()
        grad_rel_diff = ((grad_dense - grad_sparse).abs() / (grad_dense.abs() + 1e-12)).max().item()

        # Tolerances
        atol = 1e-5 if dtype == torch.float32 else 1e-10
        rtol = 1e-4 if dtype == torch.float32 else 1e-8

        score_ok = score_abs_diff <= atol or score_rel_diff <= rtol
        loss_ok = loss_abs_diff <= atol
        grad_ok = grad_abs_diff <= atol or grad_rel_diff <= rtol

        results[dtype_name] = {
            "loss_dense": float(dense_loss.item()),
            "loss_sparse": float(sparse_loss.item()),
            "loss_abs_diff": float(loss_abs_diff),
            "score_max_abs_diff": float(score_abs_diff),
            "score_max_rel_diff": float(score_rel_diff),
            "grad_max_abs_diff": float(grad_abs_diff),
            "grad_max_rel_diff": float(grad_rel_diff),
            "tolerance_atol": atol,
            "tolerance_rtol": rtol,
            "passed": bool(score_ok and loss_ok and grad_ok),
        }

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device_idx = int(os.environ.get("CUDA_DEVICE", 1 if torch.cuda.device_count() > 1 else 0))
    device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
    print(f"Running Phase C-01 exact reconstruction equivalence test on {device}...")

    res = test_reconstruction_equivalence(device=device)

    all_passed = all(r["passed"] for r in res.values())
    summary = {
        "phase": "C-01",
        "description": "Exact structural reconstruction equivalence (dense vs exact_sparse Gram)",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(device_idx) if torch.cuda.is_available() else "CPU",
        "all_passed": all_passed,
        "results": res,
    }

    json_path = OUTPUT_DIR / "exact_reconstruction_verification.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    txt_path = OUTPUT_DIR / "exact_reconstruction_verification.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Phase C-01: Exact Structural Reconstruction Equivalence ===\n")
        f.write(f"Device: {device} ({summary['gpu_name']})\n")
        f.write(f"Status: {'PASS' if all_passed else 'FAIL'}\n\n")
        for dt, r in res.items():
            f.write(f"[{dt}]\n")
            f.write(f"  Loss: dense={r['loss_dense']:.8f}, sparse={r['loss_sparse']:.8f} (diff={r['loss_abs_diff']:.2e})\n")
            f.write(f"  Score max abs diff: {r['score_max_abs_diff']:.2e}, rel diff: {r['score_max_rel_diff']:.2e}\n")
            f.write(f"  Grad max abs diff:  {r['grad_max_abs_diff']:.2e}, rel diff: {r['grad_max_rel_diff']:.2e}\n")
            f.write(f"  Passed: {r['passed']}\n")

    print(f"Results written to:\n  {json_path}\n  {txt_path}")
    for dt, r in res.items():
        print(f"[{dt}] Score abs diff: {r['score_max_abs_diff']:.2e}, Grad abs diff: {r['grad_max_abs_diff']:.2e} -> {'PASS' if r['passed'] else 'FAIL'}")

    if not all_passed:
        print("CRITICAL: Exact reconstruction equivalence failed!")
        sys.exit(1)
    else:
        print("PHASE C-01 EXACT RECONSTRUCTION EQUIVALENCE PASSED!")


if __name__ == "__main__":
    main()
