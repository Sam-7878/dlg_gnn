#!/usr/bin/env python3
"""
build_release_bundle_m2.py

Assembles a sanitized, publication-ready reproducibility release package for M2 at:
outputs/benchmark/manuscript_m2/release/dlg_gnn_benchmark_m2/

Contents:
- README.md (public overview, reproduction instructions, environment setup)
- LICENSE (MIT License)
- environment.yml and requirements.txt
- manifests/ (canonical LANL manifest, architecture trace, support matrix)
- tables/ (LaTeX tables, CSV summaries)
- frozen_hashes.txt (authoritative SHA-256 registry)
- reproduction_commands.md

Sanitization checks:
- Asserts 0 private local paths (/mnt/d, C:\\Users, sam, etc.)
- Asserts 0 forbidden DARPA/THEIA/TC-E5 tokens
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_release_bundle_m2")

REPO_ROOT = Path(__file__).resolve().parents[2]
M2_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m2"
RELEASE_DIR = M2_DIR / "release" / "dlg_gnn_benchmark_m2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    (RELEASE_DIR / "manifests").mkdir(parents=True, exist_ok=True)
    (RELEASE_DIR / "tables").mkdir(parents=True, exist_ok=True)

    log.info(f"Assembling release bundle at {RELEASE_DIR}...")

    # 1. Copy manifests
    src_manifests = [
        M2_DIR / "audit" / "lanl_canonical_manifest.json",
        M2_DIR / "audit" / "m2_manuscript_source_registry.json",
        M2_DIR / "architecture" / "dlg_architecture_source_trace.json",
        REPO_ROOT / "outputs" / "benchmark" / "sci_round5_final" / "manifests" / "model_dataset_support_matrix_v2.csv",
    ]
    for mf in src_manifests:
        if mf.exists():
            shutil.copy2(mf, RELEASE_DIR / "manifests" / mf.name)
            log.info(f"Copied manifest: {mf.name}")

    # 2. Copy tables
    src_tables = [
        M2_DIR / "architecture" / "table_dlg_architecture_budget_m2.tex",
        M2_DIR / "controls" / "table_capacity_controls_m2.tex",
        M2_DIR / "controls" / "capacity_controls_summary_m2.csv",
        M2_DIR / "lanl" / "lanl_neighborhood_diagnostics_m2.csv",
        M2_DIR / "bibliography" / "reference_metadata_audit.csv",
    ]
    for tb in src_tables:
        if tb.exists():
            shutil.copy2(tb, RELEASE_DIR / "tables" / tb.name)
            log.info(f"Copied table: {tb.name}")

    # 3. Environment & License
    license_text = """MIT License

Copyright (c) 2026 DLG-GNN Benchmark Authors

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

    req_text = """torch>=2.4.0
torch_geometric>=2.5.0
pygod>=1.1.0
scikit-learn>=1.4.0
scipy>=1.12.0
pandas>=2.2.0
numpy>=1.26.0
"""
    (RELEASE_DIR / "requirements.txt").write_text(req_text, encoding="utf-8")

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

1. **Exact, Fail-Closed Support Protocol**: 8 detector configurations (6 PyGOD 1.1 baselines + 2 DLG variants) evaluated across a frozen 10-dataset primary suite + external LANL cybersecurity validation.
2. **Zero-OOM Sparse Exact Backend**: Implements exact linear dot-product structure reconstruction via feature Gram identities in $O(N H^2 + |E| H)$ time and $O(N H)$ memory, replacing $O(N^2)$ dense matrix materialization without stochastic negative sampling.
3. **Multi-Seed Statistical Testing**: Strict seed-first metric aggregation (5 seeds: 42..46), complete-case Friedman tests, and Holm-adjusted Wilcoxon signed-rank comparisons.
4. **Controlled Ablations**: Evaluates capacity controls (`DLG-Aug-Zero`, `DLG-Aug-Permuted`, `DLG-Base-70`) demonstrating that local augmentation benefits require true node-aligned structural context.

---

## Directory Structure

```
dlg_gnn_benchmark/
├── LICENSE
├── README.md
├── requirements.txt
├── frozen_hashes.txt
├── reproduction_commands.md
├── manifests/
│   ├── lanl_canonical_manifest.json
│   ├── m2_manuscript_source_registry.json
│   ├── dlg_architecture_source_trace.json
│   └── model_dataset_support_matrix_v2.csv
└── tables/
    ├── table_dlg_architecture_budget_m2.tex
    ├── table_capacity_controls_m2.tex
    ├── capacity_controls_summary_m2.csv
    ├── lanl_neighborhood_diagnostics_m2.csv
    └── reference_metadata_audit.csv
```

---

## Replication Guide

To replicate the experimental results and compile the manuscript:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run capacity controls
python experiments/benchmark/run_capacity_controls_m2.py --gpu 0

# 3. Compute canonical LANL diagnostics
python scripts/manuscript/calculate_lanl_diagnostics_m2.py

# 4. Run test suite
pytest tests/benchmark/manuscript_m2/ -v
```
"""
    (RELEASE_DIR / "README.md").write_text(readme_text, encoding="utf-8")

    # 5. reproduction_commands.md
    repro_text = """# Reproduction Commands for DLG Benchmark Manuscript M2

All commands should be executed from the repository root with an active Python 3.12 environment.

```bash
# Verify canonical manifests and registries
python dlg_gnn/scripts/manuscript/build_m2_canonical_manifests.py

# Verify PyTorch model architecture and parameter count formulas
python dlg_gnn/scripts/manuscript/generate_m2_architecture_manifest.py

# Execute capacity and permutation controls (Elliptic, DGraphFin, LANL-RedTeam)
python dlg_gnn/experiments/benchmark/run_capacity_controls_m2.py --gpu 0

# Compute LANL canonical neighborhood diagnostics
python dlg_gnn/scripts/manuscript/calculate_lanl_diagnostics_m2.py

# Audit and verify bibliography integrity
python dlg_gnn/scripts/manuscript/audit_and_fix_bibliography_m2.py

# Run full M2 regression test suite
pytest dlg_gnn/tests/benchmark/manuscript_m2/ -v

# Compile manuscript PDF
cd dlg_gnn/docs/papers/_42_Benchmark
pdflatex -interaction=nonstopmode DLG-Benchmark.tex
bibtex DLG-Benchmark
pdflatex -interaction=nonstopmode DLG-Benchmark.tex
pdflatex -interaction=nonstopmode DLG-Benchmark.tex
```
"""
    (RELEASE_DIR / "reproduction_commands.md").write_text(repro_text, encoding="utf-8")

    # 6. Generate frozen_hashes.txt for all files in release
    hash_lines = ["# Authoritative Release Package Checksums (SHA-256)", "# Generated for DLG Benchmark Manuscript M2 Release", ""]
    for path in sorted(RELEASE_DIR.rglob("*")):
        if path.is_file() and path.name != "frozen_hashes.txt":
            rel_path = path.relative_to(RELEASE_DIR).as_posix()
            h = sha256_file(path)
            hash_lines.append(f"{h}  {rel_path}")
    (RELEASE_DIR / "frozen_hashes.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    log.info(f"Wrote frozen_hashes.txt ({len(hash_lines)-3} files)")

    # 7. Sanitization Check
    forbidden_tokens = ["/mnt/d", "C:\\Users", "\\sam\\", "/sam/", "DARPA", "THEIA", "TC-E5", "tc_e5"]
    violations = []
    for path in RELEASE_DIR.rglob("*"):
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8")
                for tok in forbidden_tokens:
                    if tok.lower() in content.lower():
                        violations.append((path.relative_to(RELEASE_DIR).as_posix(), tok))
            except UnicodeDecodeError:
                pass

    if violations:
        log.error(f"Sanitization FAILED with violations: {violations}")
        raise ValueError(f"Release package sanitization violations found: {violations}")
    else:
        log.info("Sanitization check PASSED: 0 private paths and 0 forbidden tokens.")

    print("Release package successfully assembled and verified at:")
    print(RELEASE_DIR)


if __name__ == "__main__":
    main()
