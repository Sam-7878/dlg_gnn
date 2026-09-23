#!/usr/bin/env python3
"""
test_release_zip_contains_all_frozen_artifacts.py

Round P4 Publication Gate:
Verifies that DLG_GNN_Benchmark_v1.0.0_preprint.zip contains all frozen
canonical tables, manifests, and documentation required by the Preprint Data Availability claim:
- primary/benchmark_raw.csv
- primary/model_dataset_support_matrix.csv
- primary/seed_aggregated_performance.csv
- primary/rankings.csv
- primary/friedman_tests.csv
- primary/wilcoxon_holm.csv
- controls/table_capacity_controls.csv
- controls/table_capacity_controls_paired_deltas.csv
- lanl/lanl_results.csv
- manifests/canonical_m5_manifest.json
"""

from pathlib import Path
import zipfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ZIP = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_v1.0.0_preprint.zip"

REQUIRED_ZIP_PATHS = [
    "dlg_gnn/README.md",
    "dlg_gnn/INSTALL.md",
    "dlg_gnn/CITATION.cff",
    "dlg_gnn/LICENSE",
    "dlg_gnn/pyproject.toml",
    "dlg_gnn/release_metadata.json",
    "dlg_gnn/provenance/environment_manifest.json",
    "dlg_gnn/provenance/frozen_execution_environment.json",
    "dlg_gnn/provenance/current_reproduction_environment.json",
    "dlg_gnn/artifacts/primary/benchmark_raw.csv",
    "dlg_gnn/artifacts/primary/model_dataset_support_matrix.csv",
    "dlg_gnn/artifacts/primary/seed_aggregated_performance.csv",
    "dlg_gnn/artifacts/primary/rankings.csv",
    "dlg_gnn/artifacts/primary/friedman_tests.csv",
    "dlg_gnn/artifacts/primary/wilcoxon_holm.csv",
    "dlg_gnn/artifacts/controls/capacity_controls_summary_m3_raw.csv",
    "dlg_gnn/artifacts/controls/capacity_controls_paired_seed_differences_m4.csv",
    "dlg_gnn/artifacts/lanl/table_d2_lanl_external_validation.csv",
    "dlg_gnn/artifacts/lanl/lanl_neighborhood_diagnostics_m3.csv",
    "dlg_gnn/artifacts/manifests/data_freeze.json",
    "dlg_gnn/artifacts/manifests/control_reference_manifest.json",
    "dlg_gnn/scripts/reproduce_frozen_artifacts.py",
    "dlg_gnn/scripts/verify_environment.py",
    "dlg_gnn/experiments/benchmark/run_sci_round5_final.py",
    "dlg_gnn/experiments/benchmark/run_capacity_controls_m3.py",
]


def test_zip_contains_all_frozen_artifacts():
    assert RELEASE_ZIP.exists(), f"Missing {RELEASE_ZIP}"
    with zipfile.ZipFile(RELEASE_ZIP, "r") as zf:
        zip_names = set(zf.namelist())
        for req in REQUIRED_ZIP_PATHS:
            assert req in zip_names, f"Required file {req} not found in {RELEASE_ZIP.name}"
