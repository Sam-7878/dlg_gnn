#!/usr/bin/env python3
"""
build_release_bundle_m3.py

Phase F: Builds the verified public release bundle:
outputs/benchmark/manuscript_m3/release/DLG_GNN_Benchmark_M3_Release.zip

Includes:
- Complete clean code tree (src/, experiments/, scripts/, tests/, pyproject.toml)
- Canonical manifests (control_reference_manifest, lanl_canonical_manifest, m3_manuscript_source_registry)
- Architecture manifest (shared_dlg_runtime_introspection.json)
- LANL diagnostics (csv, json)
- Generated LaTeX tables
- High-quality reproduction README.md (no speculative claims, exact sparse memory specs)

Performs a clean unpack verification test in a temporary scratch directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_release_m3")

REPO_ROOT = Path(__file__).resolve().parents[2]
M3_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m3"
RELEASE_DIR = M3_DIR / "release"
ZIP_PATH = RELEASE_DIR / "DLG_GNN_Benchmark_M3_Release.zip"

README_CONTENT = """# DLG-GNN Benchmark Reproduction Package (Round M3)

[![Preprints.org Deposit Ready](https://img.shields.io/badge/Preprints.org-Deposit_Ready-blue.svg)](https://www.preprints.org/)
[![MDPI Applied Sciences Special Issue](https://img.shields.io/badge/MDPI_ApplSci-GNN_Special_Issue-green.svg)](https://www.mdpi.com/journal/applsci/special_issues/C80IXAF9V4)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![PyG 2.6+](https://img.shields.io/badge/PyG-2.6.1-red.svg)](https://pyg.org/)

This package contains the official source code, experiment runners, canonical manifests, and reproduction scripts for the manuscript:
**"Evaluating Decoupled Local-to-Global Graph Neural Networks for Unsupervised Anomaly Detection: A Standardized Benchmark"**

Target Venues:
- **Preprints.org**: Immediate open-access preprint deposit.
- **MDPI Applied Sciences**: Special Issue *"Graph Neural Networks: Theory, Methods and Applications"*.

---

## 1. System Requirements & Environment

- **OS**: Linux / Ubuntu 22.04+ (or Windows 11 WSL2)
- **Python**: 3.10+ (tested on Python 3.11 / 3.12)
- **CUDA**: 12.1 or 12.4
- **PyTorch**: 2.5.1
- **PyG (torch_geometric)**: 2.6.1

### Installation:
```bash
git clone <repository_url> dlg_gnn
cd dlg_gnn
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 2. Mathematical Formulations & Zero-OOM Implementation

The reconstruction framework evaluates the structural reconstruction loss without ever materializing an $N \\times N$ dense adjacency matrix or outer product.

### Exact Sparse Identity:
For node representations $Z \\in \\mathbb{R}^{N \\times d}$, the per-node structural reconstruction residual is:
$$\\|A_{i,:} - z_i Z^\\top\\|_2^2 = d_i - 2 z_i (Z^\\top A_{:,i}) + \\|Z\\|_F^2 \\|z_i\\|_2^2$$

where $d_i = \\sum_j A_{ij}$ is the node degree.
By precomputing $G = Z^\\top Z \\in \\mathbb{R}^{d \\times d}$ (cost $O(N d^2)$) and scattering sparse edge inner products $Z^\\top A_{:,i}$ (cost $O(E d)$), the total memory requirement is strictly $O(N d + E)$, avoiding out-of-memory errors on massive graphs ($N > 1.2M$ nodes in DGraphFin).

---

## 3. Architecture & Parameter Budgets

The decoupled local-to-global pipeline comprises two models:
- **DLG-Base**: Single-stage local-global representation learning with fused representations ($Z = \\alpha H_{\\mathrm{loc}} + (1-\\alpha) H_{\\mathrm{glob}}$).
  - Active Parameters: $129 F + 16,705$
  - Global Optimization: 50 epochs uniformly across all datasets.
- **DLG-Aug**: Two-stage decoupled representation learning where a 2-layer local GCN ($D=64$) is pretrained for 20 epochs, frozen, and concatenated to node attributes ($X_{\\mathrm{aug}} = [X \\,\\|\\, H^{\\mathrm{loc}}]$) prior to 50 epochs of global reconstruction.
  - Stage 1 Local Pretraining Parameters: $129 F + 4,224$
  - Stage 2 Global Reconstruction Parameters: $129 F + 12,480$

---

## 4. Benchmark Reproduction

### Canonical Manifest Verification:
```bash
python scripts/manuscript/build_m3_canonical_manifests.py
```

### Architecture Introspection:
```bash
python scripts/manuscript/generate_m3_architecture_manifest.py
```

### Capacity Controls Execution & Table Generation:
```bash
python experiments/benchmark/run_capacity_controls_m3.py --gpu 0
python scripts/manuscript/generate_capacity_controls_m3.py
```

### LANL Topological Diagnostics:
```bash
python scripts/manuscript/calculate_lanl_diagnostics_m3.py
```

### Automated 15-Gate Regression Suite:
```bash
pytest tests/benchmark/manuscript_m3/ -v
```

---

## 5. Artifact Directory Structure

- `src/dlg_gnn/`: Core library (detectors, exact sparse reconstruction backends, data loaders).
- `experiments/`: Multi-seed evaluation runners and sensitivity controls.
- `scripts/`: Manifest generation, LaTeX table formatters, and audit suites.
- `tests/`: End-to-end regression tests ensuring four-way triangular consistency.
- `manifests/`: Machine-readable JSON audits of all 355 frozen benchmark runs, architecture introspections, and source registries.
- `diagnostics/`: Empirical topological measurements on LANL-RedTeam.
- `generated_tables/`: Formatted LaTeX tables directly included in the manuscript.

---

## 6. License

This benchmark release is distributed under the MIT License.
"""


def create_release_bundle():
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Building release bundle at {ZIP_PATH}...")

    # Files and folders to package
    package_items = [
        # (source_path, zip_relpath)
        (REPO_ROOT / "src", "dlg_gnn_release/src"),
        (REPO_ROOT / "experiments", "dlg_gnn_release/experiments"),
        (REPO_ROOT / "scripts", "dlg_gnn_release/scripts"),
        (REPO_ROOT / "tests", "dlg_gnn_release/tests"),
        (REPO_ROOT / "pyproject.toml", "dlg_gnn_release/pyproject.toml"),
    ]

    manifests = [
        M3_DIR / "audit" / "control_reference_manifest.json",
        M3_DIR / "audit" / "lanl_canonical_manifest.json",
        M3_DIR / "audit" / "m3_manuscript_source_registry.json",
        M3_DIR / "architecture" / "shared_dlg_runtime_introspection.json",
    ]

    diagnostics = [
        M3_DIR / "lanl" / "lanl_neighborhood_diagnostics_m3.csv",
        M3_DIR / "lanl" / "lanl_neighborhood_diagnostics_m3.json",
    ]

    generated_tables_dir = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "generated"

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Add README.md
        zf.writestr("dlg_gnn_release/README.md", README_CONTENT)

        # 2. Add source code trees
        for src_path, zip_base in package_items:
            if src_path.is_file():
                zf.write(src_path, zip_base)
            elif src_path.is_dir():
                for file_path in src_path.rglob("*"):
                    if file_path.is_file():
                        # Exclude __pycache__, .pyc, logs, temp files
                        if "__pycache__" in file_path.parts or file_path.suffix in (".pyc", ".pyo", ".pyd"):
                            continue
                        rel_in_dir = file_path.relative_to(src_path)
                        target_zip_path = f"{zip_base}/{rel_in_dir.as_posix()}"
                        zf.write(file_path, target_zip_path)

        # 3. Add manifests
        for mf in manifests:
            if mf.exists():
                zf.write(mf, f"dlg_gnn_release/manifests/{mf.name}")
            else:
                log.warning(f"Manifest not found: {mf}")

        # 4. Add diagnostics
        for dg in diagnostics:
            if dg.exists():
                zf.write(dg, f"dlg_gnn_release/diagnostics/{dg.name}")
            else:
                log.warning(f"Diagnostic not found: {dg}")

        # 5. Add generated tables
        if generated_tables_dir.exists():
            for tbl in generated_tables_dir.glob("*.tex"):
                zf.write(tbl, f"dlg_gnn_release/generated_tables/{tbl.name}")

    zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    log.info(f"Successfully created release zip: {ZIP_PATH} ({zip_size_mb:.2f} MB)")


def verify_release_bundle():
    log.info("Running clean unpack verification of release bundle...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(tmp_path)
        
        extracted_root = tmp_path / "dlg_gnn_release"
        assert extracted_root.exists(), "Extracted root does not exist"
        assert (extracted_root / "src").exists(), "Missing src/ in unpacked bundle"
        assert (extracted_root / "README.md").exists(), "Missing README.md in unpacked bundle"
        assert (extracted_root / "manifests").exists(), "Missing manifests/ in unpacked bundle"

        # Test python imports from unpacked bundle
        python_cmd = sys.executable
        verify_code = (
            "import sys; "
            f"sys.path.insert(0, r'{extracted_root / 'src'}'); "
            "from gog_fraud.models.pygod.shared_reconstruction import SharedDLGBase, SharedDLGFull; "
            "print('Verified SharedDLGBase and SharedDLGFull import successfully from unpacked bundle')"
        )
        res = subprocess.run([python_cmd, "-c", verify_code], capture_output=True, text=True)
        if res.returncode != 0:
            log.error(f"Verification import failed:\n{res.stderr}")
            raise RuntimeError(f"Unpack test failed: {res.stderr}")
        log.info(f"Unpack import test output: {res.stdout.strip()}")

    print(f"Phase F Complete: Verified release bundle created at {ZIP_PATH}")


if __name__ == "__main__":
    create_release_bundle()
    verify_release_bundle()
