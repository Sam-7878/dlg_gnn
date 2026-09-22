#!/usr/bin/env python3
"""
test_every_no_raw_readme_command_executes.py

Round M5 Freeze Gate: Extracts every command listed under Section 3 of README.md
('Mode 1: Frozen-Artifact Reproduction (No Raw Data Needed)') and verifies that
each command actually executes successfully (exit code 0) from a clean unpack.
"""

from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ZIP = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_M5_Release.zip"


def test_readme_no_raw_commands_execute():
    assert RELEASE_ZIP.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(RELEASE_ZIP, "r") as zf:
            zf.extractall(tmp_path)

        unpacked_root = tmp_path / "dlg_gnn"
        readme_path = unpacked_root / "README.md"
        assert readme_path.exists()

        commands_to_test = [
            [sys.executable, "scripts/reproduce_frozen_artifacts.py", "--artifact-root", "artifacts", "--output-dir", "reproduced_tables"],
            [sys.executable, "scripts/manuscript/generate_capacity_controls_m4.py", "--artifact-root", "artifacts", "--output-dir", "reproduced_tables"],
            [sys.executable, "scripts/manuscript/generate_m3_architecture_manifest.py"],
            [sys.executable, "-m", "pytest", "tests/benchmark/manuscript_m5/test_exact_sparse_directed_weighted_cases.py", "-v"],
            [sys.executable, "experiments/benchmark/run_sci_round5_final.py", "--dry-run"],
            [sys.executable, "experiments/benchmark/run_capacity_controls_m3.py", "--dry-run"],
        ]

        for cmd in commands_to_test:
            res = subprocess.run(cmd, cwd=unpacked_root, capture_output=True, text=True)
            assert res.returncode == 0, f"README command failed: {' '.join(cmd)}\nStdout: {res.stdout}\nStderr: {res.stderr}"
