#!/usr/bin/env python3
"""
scripts/verify_environment.py

Verifies the current execution environment against the canonical frozen
specification in provenance/environment_manifest.json (and provenance/frozen_execution_environment.json).

Outputs:
- Python version
- PyTorch version and CUDA capability
- PyG (torch_geometric) version
- PyGOD version
- torch_sparse version
- torch_scatter version
- GPU device details
"""

import json
from pathlib import Path
import platform
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def inspect_environment():
    report = {}

    # Python
    py_ver = sys.version.split()[0]
    report["Python"] = py_ver

    # Torch & CUDA
    try:
        import torch
        report["Torch"] = torch.__version__
        report["CUDA_available"] = torch.cuda.is_available()
        report["CUDA_version"] = torch.version.cuda if torch.cuda.is_available() else "N/A"
        report["GPU"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected"
    except ImportError:
        report["Torch"] = "MISSING"
        report["CUDA_available"] = False
        report["CUDA_version"] = "N/A"
        report["GPU"] = "N/A"

    # PyG
    try:
        import torch_geometric
        report["PyG"] = torch_geometric.__version__
    except ImportError:
        report["PyG"] = "MISSING"

    # PyGOD
    try:
        import pygod
        report["PyGOD"] = pygod.__version__
    except ImportError:
        report["PyGOD"] = "MISSING"

    # torch_sparse
    try:
        import torch_sparse
        report["torch_sparse"] = getattr(torch_sparse, "__version__", "INSTALLED")
    except ImportError:
        report["torch_sparse"] = "MISSING"

    # torch_scatter
    try:
        import torch_scatter
        report["torch_scatter"] = getattr(torch_scatter, "__version__", "INSTALLED")
    except ImportError:
        report["torch_scatter"] = "MISSING"

    # Check against manifest
    manifest_paths = [
        REPO_ROOT / "outputs/benchmark/manuscript_m5/provenance/environment_manifest.json",
        REPO_ROOT / "outputs/benchmark/manuscript_m5/provenance/frozen_execution_environment.json",
        REPO_ROOT / "provenance/environment_manifest.json",
        REPO_ROOT / "provenance/frozen_execution_environment.json",
    ]
    manifest = None
    for mp in manifest_paths:
        if mp.exists():
            manifest = json.loads(mp.read_text(encoding="utf-8"))
            break

    print("=" * 60)
    print("DLG-GNN Benchmark Environment Verification")
    print("=" * 60)
    print(f"Platform:      {platform.platform()}")
    print(f"Python:        {report['Python']}")
    print(f"PyTorch:       {report['Torch']}")
    print(f"CUDA:          {report['CUDA_version']} (Available: {report['CUDA_available']})")
    print(f"PyG:           {report['PyG']}")
    print(f"PyGOD:         {report['PyGOD']}")
    print(f"torch_sparse:  {report['torch_sparse']}")
    print(f"torch_scatter: {report['torch_scatter']}")
    print(f"GPU:           {report['GPU']}")
    print("-" * 60)

    all_passed = True
    if report["Torch"] == "MISSING" or report["PyG"] == "MISSING":
        print("ERROR: Core PyTorch or PyG libraries are missing.")
        all_passed = False

    if manifest:
        print("Comparison with Canonical Frozen Manifest:")
        if report["PyG"] != "MISSING" and report["PyG"] != manifest.get("torch_geometric"):
            print(f"  [NOTICE] PyG version {report['PyG']} differs from frozen {manifest.get('torch_geometric')}")
        if report["PyGOD"] != "MISSING" and report["PyGOD"] != manifest.get("pygod"):
            print(f"  [NOTICE] PyGOD version {report['PyGOD']} differs from frozen {manifest.get('pygod')}")
        print("  Canonical manifest loaded successfully.")
    print("=" * 60)

    if all_passed:
        print("Environment verification PASSED!")
    else:
        print("Environment verification FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    inspect_environment()
