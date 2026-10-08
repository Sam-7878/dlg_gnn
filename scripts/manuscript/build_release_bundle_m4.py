#!/usr/bin/env python3
"""
build_release_bundle_m4.py

Round M4 Public Release Builder:
Outputs:
outputs/benchmark/manuscript_m4/release/DLG_GNN_Benchmark_M4_Release.zip

Hygiene and Scope:
- Paper-specific minimal tree (src/gog_fraud/models/pygod, core runners, manuscript scripts, manifests)
- Excludes historical DARPA/THEIA scripts, private absolute paths, unrelated modules
- Corrected exact-sparse Gram identity math and complexity separation
- Section title: 'Exact Sparse Reconstruction and Memory-Aware Execution'
- README path tree states 'src/gog_fraud/'
- Environment versions reconciled against provenance/environment_manifest.json (PyG 2.7.0, Torch 2.5.1)
- Source tree manifest with SHA-256 digests
- Automated unpack and import verification
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_release_m4")

REPO_ROOT = Path(__file__).resolve().parents[2]
M3_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m3"
M4_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m4"
RELEASE_DIR = M4_DIR / "release"
PROVENANCE_DIR = M4_DIR / "provenance"
ZIP_PATH = RELEASE_DIR / "DLG_GNN_Benchmark_M4_Release.zip"

README_CONTENT = """# DLG-GNN Benchmark Reproduction Package (Round M4)

[![Preprints.org Deposit Ready](https://img.shields.io/badge/Preprints.org-Deposit_Ready-blue.svg)](https://www.preprints.org/)
[![MDPI Applied Sciences Special Issue](https://img.shields.io/badge/MDPI_ApplSci-GNN_Special_Issue-green.svg)](https://www.mdpi.com/journal/applsci/special_issues/C80IXAF9V4)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![PyG 2.7+](https://img.shields.io/badge/PyG-2.7.0-red.svg)](https://pyg.org/)

This package contains the official source code, experiment runners, canonical manifests, and reproduction scripts for the manuscript:
**"Evaluating Decoupled Local-to-Global Graph Neural Networks for Unsupervised Anomaly Detection: A Standardized Benchmark"**

Target Venues:
- **Preprints.org**: Open-access preprint deposit.
- **MDPI Applied Sciences**: Special Issue *"Graph Neural Networks: Theory, Methods and Applications"*.

---

## 1. System Requirements & Environment

- **OS**: Linux / Ubuntu 22.04+ (or Windows 11 WSL2)
- **Python**: 3.10+ (tested on Python 3.12)
- **CUDA**: 12.1
- **PyTorch**: 2.5.1
- **PyG (torch_geometric)**: 2.7.0
- **PyGOD**: 1.1.0

See `provenance/environment_manifest.json` for full frozen package specifications.

### Installation:
```bash
git clone <repository_url> dlg_gnn
cd dlg_gnn
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 2. Exact Sparse Reconstruction and Memory-Aware Execution

The reconstruction framework evaluates the structural reconstruction loss without ever materializing an $N \\times N$ dense adjacency matrix or outer product.

### Exact Sparse Identity:
For node representations $Z \\in \\mathbb{R}^{N \\times d}$ and binary graph adjacency $A \\in \\{0, 1\\}^{N \\times N}$, the per-node structural reconstruction residual is:
$$\\|A_{i,:} - z_i Z^\\top\\|_2^2 = d_i - 2 \\sum_{j: A_{ij} \\neq 0} A_{ij} (z_i z_j^\\top) + z_i (Z^\\top Z) z_i^\\top$$

where $d_i = \\sum_j A_{ij}$ is the node degree.

### Complexity Guarantees:
- **Arithmetic Complexity**: $O(E d + N d^2)$ total operations.
- **Additional Dense Structural Storage Avoided**: $O(N^2)$ (never materialized in host or GPU memory).
- **Core Stored Graph / Embedding Memory**: $O(E + N d + d^2)$.

See `docs/math/exact_sparse_reconstruction.md` for the complete algebraic derivation and proof of non-equivalence to scalar Frobenius approximations.

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

### A. Reproducing Tables & Claims from Frozen Manifests:
These commands execute immediately without downloading massive external raw graphs:

1. **Capacity Controls & Paired-Seed Differences Table**:
   ```bash
   python scripts/manuscript/generate_capacity_controls_m4.py
   ```
2. **Architecture Introspection Verification**:
   ```bash
   python scripts/manuscript/generate_m3_architecture_manifest.py
   ```
3. **LANL Topological Diagnostics**:
   ```bash
   python scripts/manuscript/calculate_lanl_diagnostics_m3.py
   ```
4. **Canonical Manifest Integrity Verification**:
   ```bash
   python scripts/manuscript/build_m3_canonical_manifests.py
   ```
5. **Exact Sparse Mathematical Unit Test**:
   ```bash
   pytest tests/benchmark/manuscript_m4/test_release_exact_sparse_formula_matches_dense.py -v
   ```

### B. Full Re-Training from Raw Datasets (Requires Raw Corpus Downloads):
Raw datasets are subject to individual upstream licenses and repository hosting:
- Elliptic: Kaggle Anti-Money Laundering Bitcoin Dataset
- DGraphFin: FinVolution Graph Anomaly Benchmark
- LANL-RedTeam: Los Alamos National Laboratory Cyber Security Data repository

Once datasets are placed in `data/`, multi-seed experiments can be re-executed:
```bash
python experiments/benchmark/run_capacity_controls_m3.py --gpu 0
```

---

## 5. Artifact Directory Structure

- `src/gog_fraud/`: Core model library (SharedDLGBase, SharedDLGFull, GADNR, exact sparse reconstruction backends).
- `docs/math/`: Formal mathematical derivations and algebraic identities.
- `experiments/`: Multi-seed evaluation runners and sensitivity controls.
- `scripts/`: Manifest generation, LaTeX table formatters, and audit suites.
- `tests/`: Unit tests and regression suites ensuring four-way triangular consistency.
- `manifests/`: Machine-readable JSON audits of frozen primary benchmark runs, architecture introspections, and source registries.
- `diagnostics/`: Empirical topological measurements and neighbor ratio definitions on LANL-RedTeam.
- `generated_tables/`: Formatted LaTeX tables directly included in the manuscript.
- `provenance/`: Environment versions, hardware specifications, and source tree SHA-256 manifest.

---

## 6. Provenance & Integrity

- Source Tree SHA-256 Manifest: `provenance/source_tree_manifest.csv`
- Primary Benchmark Hash: `39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c`
- LANL Canonical Graph Hash: `689c2968fe3ece9494196515e6089d6db3f430530e55b8b410d116b27c920359`
- Source Git Commit: `null` (verified via full cryptographic source tree manifest)

---

## 7. License

This benchmark release is distributed under the MIT License.
"""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_release():
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Building verified minimal release bundle at {ZIP_PATH}...")

    # Define minimal paper-specific items: (source_path, zip_relpath)
    file_map: dict[str, Path] = {}

    # 1. Models: src/gog_fraud/models/pygod/
    pygod_dir = REPO_ROOT / "src" / "gog_fraud" / "models" / "pygod"
    for py_file in pygod_dir.glob("*.py"):
        file_map[f"dlg_gnn_release/src/gog_fraud/models/pygod/{py_file.name}"] = py_file

    # __init__.py files
    file_map["dlg_gnn_release/src/gog_fraud/__init__.py"] = REPO_ROOT / "src" / "gog_fraud" / "__init__.py"
    file_map["dlg_gnn_release/src/gog_fraud/models/__init__.py"] = REPO_ROOT / "src" / "gog_fraud" / "models" / "__init__.py"

    # 2. Math doc
    file_map["dlg_gnn_release/docs/math/exact_sparse_reconstruction.md"] = REPO_ROOT / "docs" / "math" / "exact_sparse_reconstruction.md"

    # 3. Experiments
    file_map["dlg_gnn_release/experiments/benchmark/run_capacity_controls_m3.py"] = REPO_ROOT / "experiments" / "benchmark" / "run_capacity_controls_m3.py"

    # 4. Scripts
    manuscript_scripts = [
        "build_m3_canonical_manifests.py",
        "generate_m3_architecture_manifest.py",
        "generate_capacity_controls_m3.py",
        "generate_capacity_controls_m4.py",
        "calculate_lanl_diagnostics_m3.py",
        "audit_and_fix_bibliography_m3.py",
    ]
    for s in manuscript_scripts:
        sp = REPO_ROOT / "scripts" / "manuscript" / s
        if sp.exists():
            file_map[f"dlg_gnn_release/scripts/manuscript/{s}"] = sp

    # 5. Tests
    m4_tests_dir = REPO_ROOT / "tests" / "benchmark" / "manuscript_m4"
    for tp in m4_tests_dir.glob("*.py"):
        file_map[f"dlg_gnn_release/tests/benchmark/manuscript_m4/{tp.name}"] = tp

    # 6. Manifests
    manifests = [
        M3_DIR / "audit" / "control_reference_manifest.json",
        M3_DIR / "audit" / "lanl_canonical_manifest.json",
        M3_DIR / "audit" / "m3_manuscript_source_registry.json",
        M3_DIR / "architecture" / "shared_dlg_runtime_introspection.json",
        M4_DIR / "release" / "exact_sparse_identity.json",
    ]
    for mf in manifests:
        if mf.exists():
            file_map[f"dlg_gnn_release/manifests/{mf.name}"] = mf

    # 7. Diagnostics
    diagnostics = [
        M3_DIR / "lanl" / "lanl_neighborhood_diagnostics_m3.csv",
        M3_DIR / "lanl" / "lanl_neighborhood_diagnostics_m3.json",
        M4_DIR / "lanl" / "lanl_neighbor_ratio_definition.json",
    ]
    for dg in diagnostics:
        if dg.exists():
            file_map[f"dlg_gnn_release/diagnostics/{dg.name}"] = dg

    # 8. Generated tables
    gen_dir = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "generated"
    for tbl in gen_dir.glob("*.tex"):
        file_map[f"dlg_gnn_release/generated_tables/{tbl.name}"] = tbl

    # 9. Top-level project files
    file_map["dlg_gnn_release/LICENSE"] = REPO_ROOT / "LICENSE"
    file_map["dlg_gnn_release/pyproject.toml"] = REPO_ROOT / "pyproject.toml"

    # 10. Provenance: environment manifest
    env_mf = PROVENANCE_DIR / "environment_manifest.json"
    file_map["dlg_gnn_release/provenance/environment_manifest.json"] = env_mf

    # Sanity checks for banned content
    banned_patterns = [
        "/mnt/d/_work/",
        "d:\\_work\\",
        "file:///d:",
        "darpa",
        "theia",
    ]
    for zip_rel, src_path in file_map.items():
        if src_path.is_file() and src_path.suffix in (".py", ".json", ".md", ".tex", ".toml", ".csv"):
            txt = src_path.read_text(encoding="utf-8", errors="ignore").lower()
            for b in banned_patterns:
                if b in txt:
                    raise AssertionError(f"Banned pattern '{b}' detected in {src_path} (mapped to {zip_rel})")

    # Generate source tree manifest
    manifest_rows = []
    for zip_rel, src_path in sorted(file_map.items()):
        file_hash = sha256_file(src_path)
        manifest_rows.append({"relative_path": zip_rel, "sha256": file_hash, "size_bytes": src_path.stat().st_size})

    # Add README and manifest itself
    readme_hash = hashlib.sha256(README_CONTENT.encode("utf-8")).hexdigest()
    manifest_rows.append({"relative_path": "dlg_gnn_release/README.md", "sha256": readme_hash, "size_bytes": len(README_CONTENT.encode("utf-8"))})

    manifest_csv_path = PROVENANCE_DIR / "source_tree_manifest.csv"
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "sha256", "size_bytes"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_sha = sha256_file(manifest_csv_path)
    (PROVENANCE_DIR / "source_tree_manifest_sha256.txt").write_text(manifest_sha + "\n", encoding="utf-8")
    log.info(f"Generated source tree manifest: {len(manifest_rows)} files (SHA: {manifest_sha[:16]}...)")

    # Add manifest files to zip map
    file_map["dlg_gnn_release/provenance/source_tree_manifest.csv"] = manifest_csv_path
    file_map["dlg_gnn_release/provenance/source_tree_manifest_sha256.txt"] = PROVENANCE_DIR / "source_tree_manifest_sha256.txt"

    # Create Zip
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write README
        zf.writestr("dlg_gnn_release/README.md", README_CONTENT)
        # Write files
        for zip_rel, src_path in file_map.items():
            zf.write(src_path, zip_rel)

    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    log.info(f"Successfully packaged M4 release zip: {ZIP_PATH} ({size_mb:.2f} MB, {len(file_map) + 1} files)")


def verify_release():
    log.info("Verifying release bundle in clean temporary directory...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(tmp_path)

        root = tmp_path / "dlg_gnn_release"
        assert root.exists()
        assert (root / "README.md").exists()
        assert (root / "src" / "gog_fraud" / "models" / "pygod" / "shared_reconstruction.py").exists()
        assert (root / "provenance" / "source_tree_manifest.csv").exists()
        assert (root / "provenance" / "environment_manifest.json").exists()
        assert (root / "generated_tables" / "table_capacity_controls_paired_deltas.tex").exists()

        # Check banned strings in unpacked files
        banned = ["/mnt/d/_work/", "d:\\_work\\", "file:///d:", "theia", "darpa"]
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".json", ".md", ".tex", ".toml", ".csv"):
                text = p.read_text(encoding="utf-8", errors="ignore").lower()
                for b in banned:
                    assert b not in text, f"Unpacked file {p.name} contains banned '{b}'"

        # Python import test from unpacked tree
        python_cmd = sys.executable
        code = (
            "import sys; "
            f"sys.path.insert(0, r'{root / 'src'}'); "
            "from gog_fraud.models.pygod.shared_reconstruction import SharedDLGBase, SharedDLGFull; "
            "print('Verified clean unpack import of SharedDLGBase and SharedDLGFull')"
        )
        res = subprocess.run([python_cmd, "-c", code], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Unpack import test failed: {res.stderr}")
        log.info(f"Unpack import test output: {res.stdout.strip()}")

    log.info("Release bundle verification PASSED!")


if __name__ == "__main__":
    build_release()
    verify_release()
