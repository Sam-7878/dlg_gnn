"""
Build and Sanitize Reproducibility Release Bundle (Phase K)
Assembles a publication-ready public release tree at dlg_gnn/release/dlg_gnn_benchmark/
Contains:
- README.md (clean paper citation placeholder, overview, reproduction guide)
- LICENSE (MIT / Academic Open Source)
- environment.yml and requirements.txt
- configs/
- manifests/
- frozen_hashes.txt (authoritative SHA-256 registry)
- reproduction_commands.md
- table_generation_commands.md
- figure_generation_commands.md
- support_matrix.csv
- dataset_provenance.md
- LANL_build_instructions.md

Sanitization checks:
- Verifies 0 private paths (/mnt/d, C:\\Users, sam, etc.)
- Verifies 0 DARPA/THEIA/TC-E5 occurrences
- Marks status: READY_FOR_PUBLIC_PUSH
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_release_bundle")

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_DIR = REPO_ROOT / "dlg_gnn/release/dlg_gnn_benchmark"
REGISTRY_JSON = REPO_ROOT / "dlg_gnn/outputs/benchmark/manuscript_m1/audit/manuscript_source_registry.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    (RELEASE_DIR / "configs").mkdir(parents=True, exist_ok=True)
    (RELEASE_DIR / "manifests").mkdir(parents=True, exist_ok=True)

    # 1. frozen_hashes.txt
    registry = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    hash_lines = ["# Authoritative Frozen Artifact Checksums (SHA-256)", "# Generated for DLG Benchmark Manuscript M1 Release", ""]
    for entry in registry:
        hash_lines.append(f"{entry['sha256']}  {entry['artifact_name']}  ({entry['role']})")
    (RELEASE_DIR / "frozen_hashes.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    # 2. support_matrix.csv
    src_support = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_round5_final/manifests/model_dataset_support_matrix_v2.csv"
    shutil.copy2(src_support, RELEASE_DIR / "support_matrix.csv")

    # 3. configs and manifests
    src_manifest = REPO_ROOT / "dlg_gnn/outputs/benchmark/manuscript_m1/architecture/dlg_architecture_manifest.json"
    if src_manifest.exists():
        shutil.copy2(src_manifest, RELEASE_DIR / "manifests/dlg_architecture_manifest.json")

    # 4. README.md
    readme_text = """# DLG-GNN Benchmark: Scalable Graph Anomaly Detection on Attributed Graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-orange.svg)](https://pytorch.org/)
[![Reproducibility: Exact](https://img.shields.io/badge/Reproducibility-Exact%20Frozen-success.svg)]()

This repository provides the official source code, evaluation manifests, and reproduction scripts for the benchmark study:

> **A Reproducible and Scalability-Aware Benchmark for Graph Anomaly Detection with Decoupled Local-to-Global GNNs**  
> SeongSu Park and Ki-Hyung Kim  
> *Applied Sciences* (Special Issue: *Graph Neural Networks: Theory, Methods and Applications*), 2026.  
> Preprint: [Preprints.org 202609.0848](https://doi.org/10.20944/preprints202609.0848.v1)

---

## Key Highlights

1. **Exact, Fail-Closed Support Protocol**: 8 detector configurations (6 PyGOD 1.1 baselines + 2 DLG variants) evaluated across a frozen 10-dataset suite + external LANL cybersecurity validation.
2. **Zero-OOM Sparse Exact Backend**: Implements exact linear dot-product structure reconstruction via feature Gram identities in $O(N H^2 + |E| H)$ time and $O(N H)$ memory, replacing $O(N^2)$ dense matrix materialization without stochastic negative sampling.
3. **Multi-Seed Statistical Testing**: Strict seed-first metric aggregation (5 seeds: 42..46), complete-case Friedman tests, and Holm-adjusted Wilcoxon signed-rank comparisons.

---

## Quickstart & Environment Setup

```bash
# Clone the repository
git clone https://github.com/goat-bank/dlg_gnn.git
cd dlg_gnn

# Create Python environment
conda env create -f release/dlg_gnn_benchmark/environment.yml
conda activate dlg_benchmark
```

## Directory Structure
- `configs/`: Model hyperparameters and benchmark configurations.
- `manifests/`: Architecture and parameter budget manifests.
- `support_matrix.csv`: Exact 8x10 model-dataset support matrix (71 supported, 9 restricted).
- `frozen_hashes.txt`: Cryptographic SHA-256 hashes of all frozen benchmark artifacts.
- `reproduction_commands.md`: CLI commands to re-run or inspect benchmark cells.
- `table_generation_commands.md`: Commands to regenerate paper tables.
- `dataset_provenance.md`: Detailed documentation of data sources and split protocols.
- `LANL_build_instructions.md`: Preprocessing steps for the LANL-RedTeam external validation graph.
"""
    (RELEASE_DIR / "README.md").write_text(readme_text, encoding="utf-8")

    # 5. LICENSE
    license_text = """MIT License

Copyright (c) 2026 DLG Benchmark Research Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    (RELEASE_DIR / "LICENSE").write_text(license_text, encoding="utf-8")

    # 6. environment.yml and requirements.txt
    env_yml = """name: dlg_benchmark
channels:
  - pytorch
  - nvidia
  - pyg
  - conda-forge
dependencies:
  - python=3.12
  - pytorch=2.5.1
  - pytorch-cuda=12.1
  - pyg=2.6.1
  - pip
  - pip:
    - pygod==1.1.0
    - torch-scatter
    - torch-sparse
    - scikit-learn>=1.4.0
    - scipy>=1.13.0
    - numpy>=1.26.0
    - pandas>=2.2.0
    - pytest>=8.0.0
"""
    (RELEASE_DIR / "environment.yml").write_text(env_yml, encoding="utf-8")

    req_txt = """torch>=2.5.0
torch-geometric>=2.6.0
pygod==1.1.0
scikit-learn>=1.4.0
scipy>=1.13.0
numpy>=1.26.0
pandas>=2.2.0
pytest>=8.0.0
"""
    (RELEASE_DIR / "requirements.txt").write_text(req_txt, encoding="utf-8")

    # 7. reproduction_commands.md
    repro_text = """# Benchmark Reproduction Commands

## 1. Verify Frozen Hashes
```bash
sha256sum -c frozen_hashes.txt
```

## 2. Generate Paper Appendix Tables
```bash
python dlg_gnn/scripts/manuscript/generate_appendix_tables.py
```

## 3. Run Capacity Controls (Phase E Sensitivity)
```bash
python dlg_gnn/experiments/benchmark/run_capacity_controls.py --datasets Elliptic,DGraphFin,LANL-RedTeam --seeds 42,43,44,45,46
```

## 4. Run Audit and Regression Tests
```bash
pytest dlg_gnn/tests/benchmark/manuscript_m1/
```
"""
    (RELEASE_DIR / "reproduction_commands.md").write_text(repro_text, encoding="utf-8")

    # 8. table_generation_commands.md & figure_generation_commands.md
    table_cmd = """# Table Generation Commands

1. **Authoritative Appendix Tables**:
   `python dlg_gnn/scripts/manuscript/generate_appendix_tables.py`
   Output: `dlg_gnn/docs/papers/_42_Benchmark/generated/appendix_performance_tables.tex`

2. **Parameter Budget Table**:
   `python dlg_gnn/scripts/manuscript/audit_dlg_architecture.py`
   Output: `dlg_gnn/docs/papers/_42_Benchmark/generated/table_dlg_architecture_budget.tex`

3. **Detector Selection Matrix**:
   `python dlg_gnn/scripts/manuscript/generate_detector_selection_audit.py`
   Output: `dlg_gnn/outputs/benchmark/manuscript_m1/detector_selection/pygod11_detector_selection_audit.csv`
"""
    (RELEASE_DIR / "table_generation_commands.md").write_text(table_cmd, encoding="utf-8")

    fig_cmd = """# Figure Generation Commands

Primary benchmark figures (ROC curves, PR curves, and resource heatmaps) are pre-generated from frozen outputs in:
`dlg_gnn/outputs/benchmark/sci_round5_final/figures/`
"""
    (RELEASE_DIR / "figure_generation_commands.md").write_text(fig_cmd, encoding="utf-8")

    # 9. dataset_provenance.md
    prov_text = """# Dataset Provenance and Preprocessing Protocols

The benchmark comprises 10 primary datasets + 1 external validation dataset:

| Dataset | Domain | Nodes $N$ | Edges $|E|$ | Feat $F$ | Class Label Provenance | Split Protocol |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Elliptic** | Bitcoin AML | 46,564 | 83,188 | 165 | Real illicit vs licit transaction labels | Stratified node-transductive (60/20/20) |
| **DGraphFin** | Finance | 1,225,601 | 746,271 | 17 | Real loan fraud labels | Official 70/15/15 train/val/test split |
| **Yelp-Syn** | Social / Reviews | 716,847 | 14,004,995 | 300 | Synthetic feature/structural injection | Stratified node-transductive (60/20/20) |
| **Amazon-Syn** | E-commerce | 13,752 | 507,378 | 767 | Synthetic injection | Stratified node-transductive (60/20/20) |
| **Flickr-Syn** | Social Image | 89,250 | 1,001,494 | 500 | Synthetic injection | Stratified node-transductive (60/20/20) |
| **Reddit-Syn** | Discussion | 232,965 | 114,869,737 | 602 | Synthetic injection | Stratified node-transductive (60/20/20) |
| **Cora-Syn** | Citation | 2,708 | 11,276 | 1,433 | Synthetic injection | Stratified node-transductive (60/20/20) |
| **CiteSeer-Syn**| Citation | 3,327 | 9,914 | 3,703 | Synthetic injection | Stratified node-transductive (60/20/20) |
| **PubMed-Syn**  | Citation | 19,717 | 93,958 | 500 | Synthetic injection | Stratified node-transductive (60/20/20) |
| **LANL-RedTeam**| Cyber Host Log| 16,694 | 323,897 | 17 | Real red-team compromise labels | Stratified 5-seed evaluation |
"""
    (RELEASE_DIR / "dataset_provenance.md").write_text(prov_text, encoding="utf-8")

    # 10. LANL_build_instructions.md
    lanl_inst = """# LANL-RedTeam Graph Construction Protocol

The LANL cybersecurity dataset is sourced from the Los Alamos National Laboratory Cyber Security Data (Kent, 2015).

1. **Authentication Events**: Extracted from `auth.txt.gz` for Day 1 to Day 30.
2. **Graph Formulation**: Nodes represent unique computer host entities and user accounts; directed edges represent authentication flows with edge attributes capturing authentication type, logon orientation, and frequency.
3. **Ground Truth**: Ground-truth compromised nodes are derived strictly from `redteam.txt.gz`.
"""
    (RELEASE_DIR / "LANL_build_instructions.md").write_text(lanl_inst, encoding="utf-8")

    # Sanitization checks
    log.info("Running sanitization audit on release bundle...")
    banned_terms = ["/mnt/d", "C:\\Users", "sam\\", "DARPA", "THEIA", "TC-E5"]
    violations = []
    for p in RELEASE_DIR.rglob("*"):
        if p.is_file() and p.suffix in [".md", ".txt", ".yml", ".json", ".csv"]:
            content = p.read_text(encoding="utf-8", errors="ignore")
            for term in banned_terms:
                if term in content:
                    violations.append((p.name, term))

    if violations:
        raise ValueError(f"Sanitization violations detected: {violations}")

    log.info("Sanitization checks PASSED (0 private paths, 0 DARPA/THEIA references).")
    log.info("Release bundle state: READY_FOR_PUBLIC_PUSH")


if __name__ == "__main__":
    main()
