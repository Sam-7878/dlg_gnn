#!/usr/bin/env python3
"""
test_environment_manifest_matches_install_docs.py

Round M5 Unit Test: Verifies that installation documents and dependency files
strictly match the frozen specifications in environment_manifest.json.
"""

import json
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_environment_manifest_consistency():
    manifest_path = REPO_ROOT / "outputs/benchmark/manuscript_m5/provenance/environment_manifest.json"
    assert manifest_path.exists(), f"Missing environment manifest: {manifest_path}"
    env_mf = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. Check requirements-cuda121.txt
    req_cuda = REPO_ROOT / "requirements-cuda121.txt"
    assert req_cuda.exists(), f"Missing {req_cuda}"
    cuda_txt = req_cuda.read_text(encoding="utf-8")
    assert f"torch=={env_mf['torch']}" in cuda_txt
    assert f"torch-geometric=={env_mf['torch_geometric']}" in cuda_txt
    assert f"torch-scatter=={env_mf['torch_scatter']}" in cuda_txt
    assert f"torch-sparse=={env_mf['torch_sparse']}" in cuda_txt
    assert f"pygod=={env_mf['pygod']}" in cuda_txt

    # 2. Check INSTALL.md
    install_md = REPO_ROOT / "INSTALL.md"
    assert install_md.exists(), f"Missing {install_md}"
    install_txt = install_md.read_text(encoding="utf-8")
    assert "CUDA 12.1" in install_txt
    assert "PyTorch 2.5.1" in install_txt
    assert "torch-geometric==2.7.0" in install_txt
    assert "pygod==1.1.0" in install_txt

    # 3. Check environment.yml
    env_yml = REPO_ROOT / "environment.yml"
    assert env_yml.exists(), f"Missing {env_yml}"
    yml_data = yaml.safe_load(env_yml.read_text(encoding="utf-8"))
    deps = yml_data.get("dependencies", [])
    assert any("pytorch=2.5.1" in str(d) for d in deps)
    assert any("pytorch-cuda=12.1" in str(d) for d in deps)
