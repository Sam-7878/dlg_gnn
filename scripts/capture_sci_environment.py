from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(); root = Path(args.repo_root).resolve()
    packages = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions() if dist.metadata.get("Name")
    )
    (root / "requirements-sci-lock.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    environment = {"os": platform.platform(), "python": platform.python_version(), "packages": {}}
    for name in ("torch", "torch-geometric", "pygod", "numpy", "pandas", "scikit-learn", "psutil"):
        try: environment["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: environment["packages"][name] = "NOT_INSTALLED"
    try:
        import torch
        environment["cuda"] = torch.version.cuda or "CPU_ONLY"
        environment["cuda_available"] = torch.cuda.is_available()
        environment["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
    except Exception as exc:
        environment["torch_runtime_error"] = f"{type(exc).__name__}: {exc}"
    target = root / "docs/work_reports/102_stream_mc_update2/environment_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(environment, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
