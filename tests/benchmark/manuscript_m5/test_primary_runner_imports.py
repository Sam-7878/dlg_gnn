#!/usr/bin/env python3
"""
test_primary_runner_imports.py

Round M5 Freeze Gate: Verifies that the primary benchmark runner
(experiments/benchmark/run_sci_round5_final.py) imports without dependency errors
and completes its dry-run schedule verification.
"""

from pathlib import Path
import subprocess
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "experiments/benchmark/run_sci_round5_final.py"


def test_primary_runner_help():
    assert RUNNER.exists()
    res = subprocess.run([sys.executable, str(RUNNER), "--help"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"Primary runner --help failed: {res.stderr}"
    assert "Primary 10-Dataset DLG Benchmark Runner" in res.stdout


def test_primary_runner_dry_run():
    assert RUNNER.exists()
    res = subprocess.run([sys.executable, str(RUNNER), "--dry-run"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"Primary runner --dry-run failed: {res.stderr}"
    combined = res.stdout + "\n" + res.stderr
    assert "PRIMARY BENCHMARK RUNNER (DRY-RUN VALIDATION)" in combined
    assert "Datasets (10):" in combined
    assert "Models (8):" in combined
    assert "Primary runner dry-run validation PASSED with 0 errors" in combined
