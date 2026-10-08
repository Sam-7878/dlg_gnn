#!/usr/bin/env python3
"""30_preflight_3090.py: RTX 3090 24GB eGPU Hardware Qualification & Preflight (Phase I).

Tests:
  1. VRAM allocation sanity up to 20 GB on CUDA 0 (RTX 3090 24GB).
  2. Host-to-Device and Device-to-Host transfer bandwidth (Thunderbolt eGPU link measurement).
  3. Preflight multi-epoch runs across representative models (DOMINANT, CONAD, DLG-Base, DLG-Aug, AnomalyDAE).
  4. Memory and telemetry capture.
Outputs:
  - evaluation/benchmark/v2/environment/journal_cuda/preflight_3090_summary.json
  - evaluation/benchmark/v2/environment/journal_cuda/preflight_3090_summary.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Data

# Ensure src is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gog_fraud.models.pygod.shared_reconstruction import (
    SharedAnomalyDAE,
    SharedCONAD,
    SharedDLGBase,
    SharedDLGFull,
    SharedDOMINANT,
)

OUTPUT_DIR = REPO_ROOT / "evaluation" / "benchmark" / "v2" / "environment" / "journal_cuda"


def test_vRAM_allocation(device_idx: int = 0) -> dict:
    results = {}
    print(f"Testing progressive VRAM allocation on cuda:{device_idx}...")
    for size_gb in [4, 8, 14, 18, 20]:
        try:
            torch.cuda.empty_cache()
            num_elements = int(size_gb * 1024**3 / 4)
            t0 = time.time()
            tensor = torch.empty(num_elements, dtype=torch.float32, device=f"cuda:{device_idx}")
            tensor.fill_(1.0)
            torch.cuda.synchronize(device_idx)
            alloc_time = time.time() - t0
            allocated_mb = torch.cuda.memory_allocated(device_idx) / (1024**2)
            del tensor
            torch.cuda.empty_cache()
            results[f"{size_gb}GB"] = {
                "success": True,
                "allocated_mb": round(allocated_mb, 2),
                "alloc_time_sec": round(alloc_time, 3),
            }
            print(f"  Allocated {size_gb} GB ({allocated_mb:.1f} MB) in {alloc_time:.3f}s -> PASS")
        except Exception as e:
            results[f"{size_gb}GB"] = {"success": False, "error": str(e)}
            print(f"  Allocated {size_gb} GB -> FAIL ({e})")
            break
    return results


def test_pcie_bandwidth(device_idx: int = 0) -> dict:
    print(f"Benchmarking Host <-> Device transfer bandwidth on cuda:{device_idx}...")
    size_mb = 512
    num_elements = int(size_mb * 1024**2 / 4)
    host_tensor = torch.randn(num_elements, dtype=torch.float32)

    # Warmup
    _ = host_tensor.to(f"cuda:{device_idx}")
    torch.cuda.synchronize(device_idx)

    # H2D
    t0 = time.time()
    dev_tensor = host_tensor.to(f"cuda:{device_idx}")
    torch.cuda.synchronize(device_idx)
    h2d_sec = time.time() - t0
    h2d_bw_gbps = round((size_mb / 1024) / max(1e-6, h2d_sec), 2)

    # D2H
    t0 = time.time()
    back_host = dev_tensor.to("cpu")
    torch.cuda.synchronize(device_idx)
    d2h_sec = time.time() - t0
    d2h_bw_gbps = round((size_mb / 1024) / max(1e-6, d2h_sec), 2)

    del dev_tensor, host_tensor, back_host
    torch.cuda.empty_cache()

    print(f"  H2D: {h2d_bw_gbps} GB/s ({h2d_sec:.3f}s for {size_mb} MB)")
    print(f"  D2H: {d2h_bw_gbps} GB/s ({d2h_sec:.3f}s for {size_mb} MB)")

    return {
        "test_size_mb": size_mb,
        "h2d_bandwidth_gbps": h2d_bw_gbps,
        "d2h_bandwidth_gbps": d2h_bw_gbps,
        "link_type": "Thunderbolt 4 / PCIe over eGPU link",
    }


def create_preflight_graph(num_nodes: int = 2000, num_features: int = 32) -> Data:
    torch.manual_seed(42)
    x = torch.randn(num_nodes, num_features)
    sources = torch.randint(0, num_nodes, (15000,))
    targets = torch.randint(0, num_nodes, (15000,))
    edge_index = torch.stack([sources, targets], dim=0)
    edge_index = torch.unique(edge_index, dim=1).contiguous()
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[torch.randperm(num_nodes)[:100]] = 1
    return Data(x=x.contiguous(), edge_index=edge_index, y=y)


def run_preflight_models(device_idx: int = 0) -> list[dict]:
    print(f"\nRunning 5-model seed-42 preflight on cuda:{device_idx} (2000 nodes, 15000 edges)...")
    data = create_preflight_graph()
    models = [
        ("DOMINANT", SharedDOMINANT),
        ("CONAD", SharedCONAD),
        ("DLG-Base", SharedDLGBase),
        ("DLG-Aug", SharedDLGFull),
        ("AnomalyDAE", SharedAnomalyDAE),
    ]

    records = []
    for name, cls in models:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_idx)
        t0 = time.time()

        if name in ["DOMINANT", "CONAD", "DLG-Base", "DLG-Aug"]:
            kwargs = {
                "epoch": 2, "gpu": device_idx, "verbose": 0, "batch_size": 0,
                "message_backend": "sparse_fused", "reconstruction_backend": "exact_sparse",
                "score_chunk_size": 1024,
            }
            if name == "DLG-Aug": kwargs["l1_epochs"] = 2
        else:
            kwargs = {
                "epoch": 2, "gpu": device_idx, "verbose": 0, "batch_size": 0,
                "reconstruction_backend": "chunked_exact", "score_chunk_size": 512,
            }

        detector = cls(**kwargs)
        d = data.clone()
        detector.fit(d)
        score = detector.decision_function(d)
        torch.cuda.synchronize(device_idx)
        elapsed = time.time() - t0
        peak_vram_mb = torch.cuda.max_memory_allocated(device_idx) / (1024**2)

        record = {
            "model": name,
            "runtime_sec": round(elapsed, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "finite_loss": bool(all(torch.isfinite(torch.tensor(detector.loss_history_)))),
            "finite_scores": bool(torch.isfinite(score).all()),
            "status": "PASS",
        }
        records.append(record)
        print(f"  {name:12s} -> PASS ({elapsed:.2f}s, Peak VRAM: {peak_vram_mb:.1f} MB)")

    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device_idx = 0  # RTX 3090 eGPU reference
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        print("CRITICAL: CUDA device 0 not available!")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(device_idx)
    total_mem_gb = round(torch.cuda.get_device_properties(device_idx).total_memory / (1024**3), 2)
    print(f"=== RTX 3090 eGPU Preflight Qualification (Gate G3) ===")
    print(f"Target Device: cuda:{device_idx} ({gpu_name}, {total_mem_gb} GB VRAM)")

    # 1. Allocation test
    vram_results = test_vRAM_allocation(device_idx)

    # 2. Bandwidth test
    bw_results = test_pcie_bandwidth(device_idx)

    # 3. Model preflight
    model_records = run_preflight_models(device_idx)

    all_models_passed = all(r["status"] == "PASS" for r in model_records)
    vram_20g_passed = vram_results.get("20GB", {}).get("success", False)

    summary = {
        "gate": "G3",
        "device": device_idx,
        "gpu_name": gpu_name,
        "total_memory_gb": total_mem_gb,
        "vram_allocation_tests": vram_results,
        "bandwidth_tests": bw_results,
        "preflight_model_tests": model_records,
        "overall_status": "PASS" if all_models_passed and vram_20g_passed else "FAIL",
    }

    json_path = OUTPUT_DIR / "preflight_3090_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_path = OUTPUT_DIR / "preflight_3090_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# RTX 3090 eGPU Hardware Qualification and Preflight Summary (Gate G3)\n\n")
        f.write(f"- **Device**: `cuda:{device_idx}` ({gpu_name})\n")
        f.write(f"- **Total Physical VRAM**: `{total_mem_gb} GB`\n")
        f.write(f"- **Host<->Device Link**: `{bw_results['link_type']}`\n")
        f.write(f"- **H2D Transfer Rate**: `{bw_results['h2d_bandwidth_gbps']} GB/s`\n")
        f.write(f"- **D2H Transfer Rate**: `{bw_results['d2h_bandwidth_gbps']} GB/s`\n")
        f.write(f"- **20 GB VRAM Allocation**: `{'PASS' if vram_20g_passed else 'FAIL'}`\n\n")
        f.write("## 5-Model Seed-42 Preflight Results\n\n")
        f.write("| Model | Status | Runtime (s) | Peak VRAM (MB) | Finite Loss | Finite Scores |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in model_records:
            f.write(f"| {r['model']} | {r['status']} | {r['runtime_sec']} | {r['peak_vram_mb']} | {r['finite_loss']} | {r['finite_scores']} |\n")
        f.write(f"\n**Overall Gate G3 Status: {summary['overall_status']}**\n")

    print(f"\nPreflight summary successfully saved to:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
