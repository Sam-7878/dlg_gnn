"""Export hyperparameters and reproducibility environment for SCI-v3 R2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from validation.sci_v3_final_common import atomic_csv, atomic_json, sha256_file


def flatten(value: Any, prefix: str = "") -> list[dict[str, str]]:
    rows = []
    if isinstance(value, dict):
        for key, item in value.items(): rows.extend(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        rows.append({"parameter": prefix, "value": json.dumps(value)})
    else:
        rows.append({"parameter": prefix, "value": str(value)})
    return rows


def command(*args: str) -> str:
    try: return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except Exception as exc: return f"unavailable: {type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sci_v3_submission_r2/closure.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_path = Path(cfg["base_config"]); base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    root = Path(cfg["output_root"]); destination = root / "reproducibility"; destination.mkdir(parents=True, exist_ok=True)
    parameters = pd.DataFrame(flatten({"submission_r2": cfg, "production_closure": base}))
    atomic_csv(destination / "hyperparameters.csv", parameters)
    destination.joinpath("hyperparameters.tex").write_text(parameters.to_latex(index=False, escape=True), encoding="utf-8")
    package_lines = command(sys.executable, "-m", "pip", "freeze").splitlines()
    checkpoints = {}
    for seed in cfg["seeds"]:
        for name in ("level1.pt", "level2.pt", "tabular.joblib", "metadata.json"):
            path = Path(cfg["source_results"]) / f"checkpoints/seed{seed}/{name}"
            checkpoints[str(path)] = sha256_file(path)
    environment = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "kernel": platform.release(),
        "cpu": platform.processor(),
        "memory_limit_note": "WSL2 configured by user for 20 GiB of a 32 GiB host",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_dirty": bool(command("git", "status", "--porcelain")),
        "config_hashes": {str(args.config): sha256_file(args.config), str(base_path): sha256_file(base_path)},
        "checkpoint_hashes": checkpoints,
        "package_freeze_sha256": hashlib.sha256("\n".join(package_lines).encode()).hexdigest(),
        "environment_variables_recorded": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG")},
    }
    atomic_json(destination / "environment.json", environment)
    destination.joinpath("requirements_freeze.txt").write_text("\n".join(package_lines) + "\n", encoding="utf-8")
    print(json.dumps({"parameters": len(parameters), "packages": len(package_lines), "gpu": environment["gpu"]}, indent=2))


if __name__ == "__main__":
    main()
