#!/usr/bin/env python3
"""00_environment_probe.py: Comprehensive system and accelerator environment probe for Work Order A02."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "evaluation" / "benchmark" / "v2" / "environment" / "journal_cuda"


def run_cmd(cmd: list[str]) -> str:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return f"UNAVAILABLE: {e}"


def get_meminfo() -> dict:
    info = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    k, v = parts[0].strip(), parts[1].strip()
                    if k in ("MemTotal", "MemFree", "MemAvailable", "SwapTotal", "SwapFree"):
                        info[k] = v
    except Exception as e:
        info["error"] = str(e)
    return info


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    probe = {}

    # Python info
    probe["python"] = {
        "executable": sys.executable,
        "version": sys.version,
        "version_info": list(sys.version_info),
        "compiler": platform.python_compiler(),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "platform": platform.platform(),
    }

    # OS / WSL / Kernel
    probe["os"] = {
        "uname": run_cmd(["uname", "-a"]),
        "glibc": platform.libc_ver(),
        "cpu_model": run_cmd(["bash", "-c", "lscpu | grep 'Model name' | head -1"]),
        "meminfo": get_meminfo(),
    }

    # Build tools
    tools = {}
    for tool in ["pip", "setuptools", "wheel", "cmake", "ninja", "gcc", "g++", "clang", "nvcc", "hipcc"]:
        if tool in ["pip", "setuptools", "wheel"]:
            try:
                mod = __import__(tool)
                tools[tool] = getattr(mod, "__version__", "INSTALLED")
            except ImportError:
                tools[tool] = "NOT_INSTALLED"
        else:
            tools[tool] = run_cmd(["which", tool])
    probe["build_tools"] = tools

    # PyTorch and Accelerators
    import torch
    probe["torch"] = {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hip_version": torch.version.hip,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "backend": "ROCm" if torch.version.hip else ("CUDA" if torch.version.cuda else "CPU"),
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI if hasattr(torch._C, "_GLIBCXX_USE_CXX11_ABI") else "UNKNOWN",
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32 if hasattr(torch.backends, "cuda") else None,
        "tf32_cudnn_allowed": torch.backends.cudnn.allow_tf32 if hasattr(torch.backends, "cudnn") else None,
    }

    # GPU details
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info = {
                "index": i,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "major": props.major,
                "minor": props.minor,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
            gpus.append(gpu_info)
    probe["gpus"] = gpus

    # nvidia-smi probe
    probe["nvidia_smi"] = run_cmd(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version", "--format=csv"])

    # Installed packages
    import importlib.metadata
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    probe["packages"] = {k: packages[k] for k in sorted(packages.keys())}

    # Save outputs
    json_path = OUTPUT_DIR / "environment_probe.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(probe, f, indent=2)

    txt_path = OUTPUT_DIR / "environment_probe.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== Environment Probe for Work Order A02 ===\n")
        f.write(f"Backend: {probe['torch']['backend']}\n")
        f.write(f"Python: {probe['python']['version']}\n")
        f.write(f"PyTorch: {probe['torch']['version']} (CUDA: {probe['torch']['cuda_version']})\n")
        f.write(f"GPUs detected ({len(gpus)}):\n")
        for g in gpus:
            f.write(f"  [{g['index']}] {g['name']} ({g['total_memory_gb']} GB, CC {g['compute_capability']})\n")
        f.write(f"\nKey packages:\n")
        for pkg in ["torch", "torchvision", "torchaudio", "torch-geometric", "pygod", "scikit-learn", "numpy", "scipy", "pandas"]:
            f.write(f"  {pkg}: {packages.get(pkg, 'NOT_INSTALLED')}\n")

    print(f"Environment probe successfully written to:\n  {json_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
