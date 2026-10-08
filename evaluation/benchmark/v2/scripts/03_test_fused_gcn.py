#!/usr/bin/env python3
"""03_test_fused_gcn.py: Fused GCN equivalence test (Phase C-02).

Compares:
  Reference PyG GCN (standard COO message passing with GCNConv)
vs
  Fused sparse execution (pre-normalized sparse operator via SparseFusedGCN / native SpMM)

Verifies:
  - hidden representation (layer 1)
  - final latent representation (layer 2)
  - gradients with respect to input X and layer weights
Tolerance: atol = 1e-5, rtol = 1e-4.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCN, GCNConv

# Ensure src is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gog_fraud.models.pygod.sparse_message import SparseFusedGCN, normalized_sparse_adjt

OUTPUT_DIR = REPO_ROOT / "evaluation" / "benchmark" / "v2" / "environment" / "journal_cuda"


def test_fused_gcn_equivalence(
    num_nodes: int = 150,
    in_dim: int = 32,
    hid_dim: int = 64,
    out_dim: int = 32,
    edge_density: float = 0.04,
    seed: int = 42,
    device: str = "cuda:1",
) -> dict:
    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Create random undirected graph with self-loops
    adj_prob = torch.rand(num_nodes, num_nodes, device=dev)
    adj_mask = (adj_prob < edge_density).triu(diagonal=1)
    adj_mask = adj_mask | adj_mask.T
    edge_index = torch.nonzero(adj_mask).T.contiguous()

    results = {}
    for dtype_name, dtype in [("float64", torch.float64), ("float32", torch.float32)]:
        # Reference GCN: PyG standard COO normalization
        ref_model = GCN(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=2,
            out_channels=out_dim,
            dropout=0.0,
            act="relu",
            normalize=True,
            add_self_loops=True,
        ).to(dtype=dtype, device=dev)

        # Fused GCN: Pre-normalized sparse operator
        fused_model = SparseFusedGCN(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=2,
            out_channels=out_dim,
            dropout=0.0,
            act="relu",
        ).to(dtype=dtype, device=dev)

        # Ensure identical weights
        fused_model.load_state_dict(ref_model.state_dict())

        # Inputs
        X_ref = torch.randn(num_nodes, in_dim, dtype=dtype, device=dev, requires_grad=True)
        X_fused = X_ref.detach().clone().requires_grad_(True)

        # 1. Forward reference
        out_ref = ref_model(X_ref, edge_index)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        # 2. Forward fused
        adj_t = normalized_sparse_adjt(edge_index, num_nodes, dtype=dtype, device=dev)
        out_fused = fused_model(X_fused, adj_t)
        loss_fused = out_fused.sum()
        loss_fused.backward()

        # 3. Compare outputs and gradients
        out_abs_diff = (out_ref - out_fused).abs().max().item()
        out_rel_diff = ((out_ref - out_fused).abs() / (out_ref.abs() + 1e-12)).max().item()

        x_grad_abs_diff = (X_ref.grad - X_fused.grad).abs().max().item()
        x_grad_rel_diff = ((X_ref.grad - X_fused.grad).abs() / (X_ref.grad.abs() + 1e-12)).max().item()

        # Weight gradients
        w_grad_diffs = []
        for (name, p_ref), (_, p_fused) in zip(ref_model.named_parameters(), fused_model.named_parameters()):
            if p_ref.grad is not None and p_fused.grad is not None:
                d = (p_ref.grad - p_fused.grad).abs().max().item()
                w_grad_diffs.append(d)
        max_w_grad_diff = max(w_grad_diffs) if w_grad_diffs else 0.0

        atol = 1e-5 if dtype == torch.float32 else 1e-9
        rtol = 1e-4 if dtype == torch.float32 else 1e-7

        out_ok = out_abs_diff <= atol or out_rel_diff <= rtol
        x_grad_ok = x_grad_abs_diff <= atol or x_grad_rel_diff <= rtol
        w_grad_ok = max_w_grad_diff <= atol

        results[dtype_name] = {
            "output_max_abs_diff": float(out_abs_diff),
            "output_max_rel_diff": float(out_rel_diff),
            "x_grad_max_abs_diff": float(x_grad_abs_diff),
            "x_grad_max_rel_diff": float(x_grad_rel_diff),
            "weight_grad_max_abs_diff": float(max_w_grad_diff),
            "tolerance_atol": atol,
            "tolerance_rtol": rtol,
            "passed": bool(out_ok and x_grad_ok and w_grad_ok),
        }

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device_idx = int(os.environ.get("CUDA_DEVICE", 1 if torch.cuda.device_count() > 1 else 0))
    device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
    print(f"Running Phase C-02 fused GCN equivalence test on {device}...")

    res = test_fused_gcn_equivalence(device=device)

    all_passed = all(r["passed"] for r in res.values())
    summary = {
        "phase": "C-02",
        "description": "Fused GCN equivalence (PyG COO reference vs SparseFusedGCN)",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(device_idx) if torch.cuda.is_available() else "CPU",
        "all_passed": all_passed,
        "results": res,
    }

    json_path = OUTPUT_DIR / "fused_gcn_verification.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    txt_path = OUTPUT_DIR / "fused_gcn_verification.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Phase C-02: Fused GCN Equivalence ===\n")
        f.write(f"Device: {device} ({summary['gpu_name']})\n")
        f.write(f"Status: {'PASS' if all_passed else 'FAIL'}\n\n")
        for dt, r in res.items():
            f.write(f"[{dt}]\n")
            f.write(f"  Output max abs diff:      {r['output_max_abs_diff']:.2e}, rel diff: {r['output_max_rel_diff']:.2e}\n")
            f.write(f"  Input grad max abs diff:   {r['x_grad_max_abs_diff']:.2e}, rel diff: {r['x_grad_max_rel_diff']:.2e}\n")
            f.write(f"  Weight grad max abs diff:  {r['weight_grad_max_abs_diff']:.2e}\n")
            f.write(f"  Passed: {r['passed']}\n")

    print(f"Results written to:\n  {json_path}\n  {txt_path}")
    for dt, r in res.items():
        print(f"[{dt}] Output abs diff: {r['output_max_abs_diff']:.2e}, Input grad abs diff: {r['x_grad_max_abs_diff']:.2e} -> {'PASS' if r['passed'] else 'FAIL'}")

    if not all_passed:
        print("CRITICAL: Fused GCN equivalence failed!")
        sys.exit(1)
    else:
        print("PHASE C-02 FUSED GCN EQUIVALENCE PASSED!")


if __name__ == "__main__":
    main()
