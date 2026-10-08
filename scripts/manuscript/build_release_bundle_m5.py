#!/usr/bin/env python3
"""
scripts/manuscript/build_release_bundle_m5.py

Round M5 Public Release Builder:
Outputs:
outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_M5_Release.zip

Key Enhancements in Round M5:
- Two-Tier Reproduction Structure: Mode 1 (frozen artifacts) and Mode 2 (full re-execution).
- Full dependency closure (src/gog_fraud/models, data, evaluation, experiments, pipelines).
- Bundles all small derived artifacts (artifacts/primary, controls, lanl, manifests).
- Canonical Mode 1 master script: scripts/reproduce_frozen_artifacts.py.
- Primary benchmark runner: experiments/benchmark/run_sci_round5_final.py.
- Sensitivity controls runner: experiments/benchmark/run_capacity_controls_m3.py.
- Unified title: 'A Reproducible and Scalability-Aware Benchmark for Graph Anomaly Detection with Decoupled Local-to-Global GNNs'.
- Environment closure: environment.yml, requirements-core.txt, requirements-cuda121.txt, INSTALL.md, scripts/verify_environment.py.
- Full automated clean unpack and reproduction verification test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_release_m5")

REPO_ROOT = Path(__file__).resolve().parents[2]
M5_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m5"
RELEASE_DIR = M5_DIR / "release"
PROVENANCE_DIR = M5_DIR / "provenance"
ART_DIR = M5_DIR / "artifacts"
ZIP_PATH = RELEASE_DIR / "DLG_GNN_Benchmark_M5_Release.zip"

README_CONTENT = """# DLG-GNN Benchmark Reproduction Package (Round M5)

[![Preprints.org Ready](https://img.shields.io/badge/Preprints.org-Preprint_Ready-blue.svg)](https://www.preprints.org/)
[![MDPI Applied Sciences](https://img.shields.io/badge/MDPI_ApplSci-GNN_Special_Issue-green.svg)](https://www.mdpi.com/journal/applsci/special_issues/C80IXAF9V4)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![PyG 2.7+](https://img.shields.io/badge/PyG-2.7.0-red.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official source code, experiment runners, canonical manifests, frozen evaluation tables, and replication scripts for the benchmark manuscript:

> **"A Reproducible and Scalability-Aware Benchmark for Graph Anomaly Detection with Decoupled Local-to-Global GNNs"**  
> *Authors*: SeongSu Park and Ki-Hyung Kim (Ajou University)  
> *Target Venues*: Preprints.org (Open-Access Preprint) and MDPI *Applied Sciences* (Special Issue: *Graph Neural Networks: Theory, Methods and Applications*)

---

## 1. System Requirements & Environment

- **OS**: Linux / Ubuntu 22.04 LTS (or Windows 11 WSL2)
- **Python**: 3.10+ (tested on Python 3.12.13)
- **CUDA**: 12.1
- **PyTorch**: 2.5.1
- **PyG (torch_geometric)**: 2.7.0
- **PyGOD**: 1.1.0

See `INSTALL.md` for step-by-step setup guides via Conda or pip.
Run `python scripts/verify_environment.py` to inspect runtime environment against `provenance/environment_manifest.json`.

---

## 2. Two-Tier Reproduction Structure

This benchmark provides two distinct reproduction pathways depending on your computational resources and dataset access:

```
┌────────────────────────────────────────────────────────────────────────┐
│ MODE 1: Frozen-Artifact Reproduction (0 Raw Datasets Required)          │
│ • Validates cryptographic hashes of frozen primary raw runs and matrix  │
│ • Regenerates all manuscript performance, statistical, & control tables │
│ • Runs exact-sparse mathematical and numerical equivalence unit tests   │
│ • Execution time: ~5 seconds (no GPU or external data downloads needed)│
└────────────────────────────────────────────────────────────────────────┘
                                    │
┌────────────────────────────────────────────────────────────────────────┐
│ MODE 2: Full Benchmark Re-Execution (Raw Datasets Required)             │
│ • Primary 10-Dataset Benchmark: run_sci_round5_final.py (355 runs)     │
│ • Sensitivity Controls: run_capacity_controls_m3.py (45 runs)          │
│ • Requires downloading external datasets according to license matrix   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mode 1: Frozen-Artifact Reproduction (No Raw Data Needed)

All small derived artifacts, seed-aggregated summaries, and machine-readable manifests are bundled in `artifacts/`.

### Master One-Line Table & Math Regeneration:
```bash
python scripts/reproduce_frozen_artifacts.py --artifact-root artifacts --output-dir reproduced_tables
```

This single command automatically executes:
1. **Hash Validation**: Verifies SHA-256 digests of `benchmark_raw.csv` (`39a497...`) and `model_dataset_support_matrix.csv` (`c58dbc...`).
2. **Primary Tables**: Generates Table 2 (Overall Performance), Table 3 (Fraud-Oriented Subset), Table 4 (Support Matrix), Table 5 (DLG Components), and Appendix Table.
3. **Statistical Tables**: Generates Average Model Rankings, Friedman test results, and Wilcoxon-Holm post-hoc tests.
4. **Capacity Controls Summary**: Formats the 3-dataset x 3-model sensitivity baseline comparison.
5. **Paired Deltas Table**: Formats Table B1 with bootstrap 95% CIs and descriptive sensitivity findings.
6. **Architecture Budget**: Generates parameter counts and layer configurations matching runtime introspection.
7. **LANL Topological Diagnostics**: Generates Table D2 neighborhood structure summary (verifying the canonical 97.27% directed in-edge ratio).
8. **Exact-Sparse Unit Tests**: Verifies exact mathematical equivalence between dense NxN reconstruction and exact-sparse Gram evaluation.

### Individual No-Raw Reproduction Commands:
You can also run individual table formatters and unit tests:
```bash
# Generate paired-seed sensitivity deltas table
python scripts/manuscript/generate_capacity_controls_m4.py --artifact-root artifacts --output-dir reproduced_tables

# Generate architecture parameter budget table
python scripts/manuscript/generate_m3_architecture_manifest.py

# Run exact-sparse numerical equivalence tests across all graph topologies
pytest tests/benchmark/manuscript_m5/test_exact_sparse_directed_weighted_cases.py -v

# Run primary runner dry-run validation
python experiments/benchmark/run_sci_round5_final.py --dry-run

# Run sensitivity controls runner dry-run validation
python experiments/benchmark/run_capacity_controls_m3.py --dry-run
```

---

## 4. Mode 2: Full Benchmark Re-Execution (Raw Datasets Required)

Full re-training requires downloading raw graphs into `data/` according to individual upstream licenses:

### Raw Dataset Acquisition Matrix:
| Dataset | Upstream Source / Access Point | Expected Location | Redistributable | Preprocessing | Canonical Hash / Version |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Elliptic** | Kaggle Anti-Money Laundering Bitcoin | `data/Elliptic/` | No | PyG standard | 46,564 nodes, 83,188 edges |
| **DGraphFin** | FinVolution / DGraph Benchmark | `data/DGraphFin/` | No | `dgraphfin_aligned.py` | 1,225,601 nodes, 746,271 edges |
| **BitcoinOTC** | SNAP Stanford Network Analysis Project | `data/BitcoinOTC/` | No | PyGOD loader | 6,005 nodes, 7,409 edges |
| **Yelp** | PyGOD / Yelp Review Anomaly Dataset | `data/Yelp/` | No | PyGOD loader | 716,847 nodes, 14,004,995 edges |
| **Amazon** | PyGOD / Amazon Co-Purchase Anomaly | `data/Amazon/` | No | PyGOD loader | 13,752 nodes, 507,378 edges |
| **Reddit** | PyGOD / Reddit Subreddit Interaction | `data/Reddit/` | No | PyGOD loader | 232,965 nodes, 114,869,737 edges |
| **Flickr** | PyGOD / Flickr User Friendship Network | `data/Flickr/` | No | PyGOD loader | 89,250 nodes, 1,001,494 edges |
| **Cora** | SNAP / Citation Graph Anomaly | `data/Cora/` | No | PyGOD loader | 2,708 nodes, 11,276 edges |
| **CiteSeer** | SNAP / Citation Graph Anomaly | `data/CiteSeer/` | No | PyGOD loader | 3,327 nodes, 9,914 edges |
| **PubMed** | SNAP / Citation Graph Anomaly | `data/PubMed/` | No | PyGOD loader | 19,717 nodes, 93,958 edges |
| **LANL-RedTeam**| Los Alamos National Laboratory Cyber Data | `data/LANL/` | No | Canonical builder | 16,694 nodes, 323,897 edges |

### A. Primary 10-Dataset Benchmark Execution:
```bash
python experiments/benchmark/run_sci_round5_final.py --config configs/benchmark/sci_round5_final.yaml --stage phase1
```

### B. Sensitivity Controls Execution (Elliptic, DGraphFin, LANL-RedTeam):
```bash
python experiments/benchmark/run_capacity_controls_m3.py --gpu 0
```

---

## 5. Exact Sparse Reconstruction Identity

For node embeddings $Z \\in \\mathbb{R}^{N \\times d}$ and adjacency matrix $A \\in \\mathbb{R}^{N \\times N}$, the structural reconstruction residual is:
$$\\|A_{i,:} - z_i Z^\\top\\|_2^2 = \\sum_{j=1}^N A_{ij}^2 - 2 \\sum_{j: A_{ij} \\neq 0} A_{ij} (z_i z_j^\\top) + z_i (Z^\\top Z) z_i^\\top$$

For binary unweighted graphs, $\\sum_j A_{ij}^2 = d_i$.
- **Arithmetic Complexity**: $O(E d + N d^2)$ total operations.
- **Dense Intermediate Memory Avoided**: $O(N^2)$ (never materialized).
- **Core Memory Required**: $O(E + N d + d^2)$.

See `docs/math/exact_sparse_reconstruction.md` for the formal algebraic derivation and proof that scalar Frobenius approximations ($\\|Z\\|_F^2 \\|z_i\\|^2$) do not hold in general.

---

## 6. Citation and Attribution

If you use this benchmark suite, please cite:

```bibtex
@article{park2026dlg_benchmark,
  title={A Reproducible and Scalability-Aware Benchmark for Graph Anomaly Detection with Decoupled Local-to-Global GNNs},
  author={Park, SeongSu and Kim, Ki-Hyung},
  journal={MDPI Applied Sciences (Under Review)},
  year={2026},
  url={https://github.com/goat-bank/dlg_gnn}
}
```

This benchmark evaluates models based on the decoupled local-to-global representation principle introduced in our preceding study:
- Park, S.; Kim, K.-H. *DLG-GNN: Decoupled Local-to-Global Graph Neural Network for Scalable Blockchain Fraud Detection*, Preprints 2026, DOI: `10.20944/preprints202609.0848.v1`.

---

## 7. License

This benchmark codebase and reproduction package are licensed under the MIT License. Upstream datasets remain subject to their respective original licenses.
"""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_release():
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Building Round M5 Verified Release Bundle at {ZIP_PATH}...")

    # File mapping: (zip_relative_path, local_source_path)
    file_map: dict[str, Path] = {}

    # 1. Models: src/gog_fraud/models/
    models_dir = REPO_ROOT / "src" / "gog_fraud" / "models"
    for py_file in models_dir.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT)
        file_map[f"dlg_gnn/{rel.as_posix()}"] = py_file

    # 2. Data modules: src/gog_fraud/data/
    data_dir = REPO_ROOT / "src" / "gog_fraud" / "data"
    for py_file in data_dir.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT)
        file_map[f"dlg_gnn/{rel.as_posix()}"] = py_file

    # 3. Evaluation modules: src/gog_fraud/evaluation/
    eval_dir = REPO_ROOT / "src" / "gog_fraud" / "evaluation"
    for py_file in eval_dir.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT)
        file_map[f"dlg_gnn/{rel.as_posix()}"] = py_file

    # 4. Experiments modules: src/gog_fraud/experiments/
    exp_dir = REPO_ROOT / "src" / "gog_fraud" / "experiments"
    for py_file in exp_dir.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT)
        file_map[f"dlg_gnn/{rel.as_posix()}"] = py_file

    # 5. Core pipelines: src/gog_fraud/pipelines/
    pipe_dir = REPO_ROOT / "src" / "gog_fraud" / "pipelines"
    for py_file in pipe_dir.glob("*.py"):
        rel = py_file.relative_to(REPO_ROOT)
        file_map[f"dlg_gnn/{rel.as_posix()}"] = py_file

    # __init__.py and analysis helper files
    file_map["dlg_gnn/src/gog_fraud/__init__.py"] = REPO_ROOT / "src" / "gog_fraud" / "__init__.py"
    file_map["dlg_gnn/src/analysis/__init__.py"] = REPO_ROOT / "src" / "analysis" / "__init__.py"
    file_map["dlg_gnn/src/analysis/utils.py"] = REPO_ROOT / "src" / "analysis" / "utils.py"

    # 6. Runners & configs
    file_map["dlg_gnn/experiments/benchmark/run_sci_round5_final.py"] = REPO_ROOT / "experiments" / "benchmark" / "run_sci_round5_final.py"
    file_map["dlg_gnn/experiments/benchmark/run_capacity_controls_m3.py"] = REPO_ROOT / "experiments" / "benchmark" / "run_capacity_controls_m3.py"
    file_map["dlg_gnn/configs/benchmark/sci_round5_final.yaml"] = REPO_ROOT / "configs" / "benchmark" / "sci_round5_final.yaml"

    # 7. Scripts
    file_map["dlg_gnn/scripts/reproduce_frozen_artifacts.py"] = REPO_ROOT / "scripts" / "reproduce_frozen_artifacts.py"
    file_map["dlg_gnn/scripts/verify_environment.py"] = REPO_ROOT / "scripts" / "verify_environment.py"
    manuscript_scripts = [
        "generate_capacity_controls_m4.py",
        "generate_m3_architecture_manifest.py",
        "calculate_lanl_diagnostics_m3.py",
        "generate_appendix_tables.py",
    ]
    for s in manuscript_scripts:
        p = REPO_ROOT / "scripts" / "manuscript" / s
        if p.exists():
            file_map[f"dlg_gnn/scripts/manuscript/{s}"] = p

    # 8. Artifacts folder (all small derived CSVs, JSONs, manifests)
    for art_file in ART_DIR.rglob("*"):
        if art_file.is_file():
            rel = art_file.relative_to(ART_DIR)
            file_map[f"dlg_gnn/artifacts/{rel.as_posix()}"] = art_file

    # 9. Math documentation
    file_map["dlg_gnn/docs/math/exact_sparse_reconstruction.md"] = REPO_ROOT / "docs" / "math" / "exact_sparse_reconstruction.md"

    # 10. Metadata, packaging, installation
    file_map["dlg_gnn/LICENSE"] = REPO_ROOT / "LICENSE"
    file_map["dlg_gnn/pyproject.toml"] = REPO_ROOT / "pyproject.toml"
    file_map["dlg_gnn/release_metadata.json"] = RELEASE_DIR / "release_metadata.json"
    file_map["dlg_gnn/CITATION.cff"] = REPO_ROOT / "CITATION.cff"
    file_map["dlg_gnn/INSTALL.md"] = REPO_ROOT / "INSTALL.md"
    file_map["dlg_gnn/environment.yml"] = REPO_ROOT / "environment.yml"
    file_map["dlg_gnn/requirements-core.txt"] = REPO_ROOT / "requirements-core.txt"
    file_map["dlg_gnn/requirements-cuda121.txt"] = REPO_ROOT / "requirements-cuda121.txt"
    file_map["dlg_gnn/provenance/environment_manifest.json"] = PROVENANCE_DIR / "environment_manifest.json"

    # 11. Tests: tests/benchmark/manuscript_m5/
    m5_tests = REPO_ROOT / "tests" / "benchmark" / "manuscript_m5"
    for tp in m5_tests.glob("*.py"):
        file_map[f"dlg_gnn/tests/benchmark/manuscript_m5/{tp.name}"] = tp

    # Sanitize check: assure zero banned patterns in packaged files
    banned_patterns = [
        "/mnt/d/_work/",
        "d:\\_work\\",
        "file:///d:",
        "darpa",
        "theia",
    ]
    for zip_rel, src_path in file_map.items():
        if src_path.is_file() and src_path.suffix in (".py", ".json", ".md", ".tex", ".toml", ".csv", ".yml", ".cff"):
            if zip_rel.endswith("test_no_private_paths.py") or zip_rel.endswith("test_no_unrelated_defense_scope.py"):
                continue
            txt = src_path.read_text(encoding="utf-8", errors="ignore").lower()
            for b in banned_patterns:
                if b in txt:
                    raise AssertionError(f"Banned pattern '{b}' detected in {src_path} (mapped to {zip_rel})")

    # Generate source tree manifest
    manifest_rows = []
    for zip_rel, src_path in sorted(file_map.items()):
        file_hash = sha256_file(src_path)
        manifest_rows.append({"relative_path": zip_rel, "sha256": file_hash, "size_bytes": src_path.stat().st_size})

    # Add README to manifest
    readme_hash = hashlib.sha256(README_CONTENT.encode("utf-8")).hexdigest()
    manifest_rows.append({"relative_path": "dlg_gnn/README.md", "sha256": readme_hash, "size_bytes": len(README_CONTENT.encode("utf-8"))})

    manifest_csv_path = PROVENANCE_DIR / "source_tree_manifest.csv"
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "sha256", "size_bytes"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_sha = sha256_file(manifest_csv_path)
    (PROVENANCE_DIR / "source_tree_manifest_sha256.txt").write_text(manifest_sha + "\n", encoding="utf-8")
    log.info(f"Generated source tree manifest: {len(manifest_rows)} files (SHA: {manifest_sha[:16]}...)")

    file_map["dlg_gnn/provenance/source_tree_manifest.csv"] = manifest_csv_path
    file_map["dlg_gnn/provenance/source_tree_manifest_sha256.txt"] = PROVENANCE_DIR / "source_tree_manifest_sha256.txt"

    # Create Zip
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dlg_gnn/README.md", README_CONTENT)
        for zip_rel, src_path in file_map.items():
            zf.write(src_path, zip_rel)

    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    log.info(f"Successfully packaged M5 Release Bundle: {ZIP_PATH} ({size_mb:.2f} MB, {len(file_map) + 1} files)")


def verify_release():
    log.info("Verifying release bundle from clean temporary directory...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(tmp_path)

        root = tmp_path / "dlg_gnn"
        assert root.exists(), "Root folder dlg_gnn missing in zip"
        assert (root / "README.md").exists(), "README.md missing"
        assert (root / "release_metadata.json").exists(), "release_metadata.json missing"
        assert (root / "CITATION.cff").exists(), "CITATION.cff missing"
        assert (root / "INSTALL.md").exists(), "INSTALL.md missing"
        assert (root / "artifacts" / "primary" / "benchmark_raw.csv").exists(), "artifacts missing"
        assert (root / "scripts" / "reproduce_frozen_artifacts.py").exists(), "reproduce script missing"
        assert (root / "experiments" / "benchmark" / "run_sci_round5_final.py").exists(), "primary runner missing"
        assert (root / "experiments" / "benchmark" / "run_capacity_controls_m3.py").exists(), "controls runner missing"

        # Check banned strings
        banned = ["/mnt/d/_work/", "d:\\_work\\", "file:///d:", "theia", "darpa"]
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".json", ".md", ".tex", ".toml", ".csv", ".yml", ".cff"):
                if p.name in ("test_no_private_paths.py", "test_no_unrelated_defense_scope.py"):
                    continue
                txt = p.read_text(encoding="utf-8", errors="ignore").lower()
                for b in banned:
                    assert b not in txt, f"Unpacked file {p.name} contains banned '{b}'"

        # Execute Mode 1 Master Reproduction from clean unpack!
        python_cmd = sys.executable
        repro_cmd = [
            python_cmd,
            str(root / "scripts" / "reproduce_frozen_artifacts.py"),
            "--artifact-root", str(root / "artifacts"),
            "--output-dir", str(root / "reproduced_tables")
        ]
        res = subprocess.run(repro_cmd, cwd=root, capture_output=True, text=True)
        if res.returncode != 0:
            log.error(f"Reproduction failed in clean unpack:\nStdout:\n{res.stdout}\nStderr:\n{res.stderr}")
            raise RuntimeError(f"Clean unpack reproduction failed with code {res.returncode}")

        log.info("Clean unpack reproduction executed with 0 errors!")
        assert (root / "reproduced_tables" / "table_appendix_all_detailed.tex").exists()
        assert (root / "reproduced_tables" / "table_capacity_controls_paired_deltas.tex").exists()
        assert (root / "reproduced_tables" / "table_lanl_diagnostics.tex").exists()

        log.info("Clean unpack release bundle verification PASSED 100%!")


if __name__ == "__main__":
    build_release()
    verify_release()
