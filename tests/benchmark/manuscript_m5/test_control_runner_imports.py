#!/usr/bin/env python3
"""
test_control_runner_imports.py

Round M5 Freeze Gate: Verifies that the sensitivity controls runner
(experiments/benchmark/run_capacity_controls_m3.py) imports without dependency errors
and completes its dry-run matrix verification.
"""

from pathlib import Path
import subprocess
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "experiments/benchmark/run_capacity_controls_m3.py"


def test_control_runner_help():
    assert RUNNER.exists()
    res = subprocess.run([sys.executable, str(RUNNER), "--help"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"Control runner --help failed: {res.stderr}"
    assert "--dry-run" in res.stdout


def test_control_runner_dry_run():
    assert RUNNER.exists()
    res = subprocess.run([sys.executable, str(RUNNER), "--dry-run"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"Control runner --dry-run failed: {res.stderr}"
    combined = res.stdout + "\n" + res.stderr
    assert "[DRY-RUN] Capacity Controls sensitivity runner dry-run mode." in combined
    assert "Total planned matrix: 45 runs." in combined
    assert "Sensitivity runner imports and arguments verified successfully." in combined
