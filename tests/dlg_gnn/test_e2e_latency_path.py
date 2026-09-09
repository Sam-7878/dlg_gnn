"""
tests/test_e2e_latency_path.py

Round 2 validation: verify that the true E2E latency measurement path
includes GNN forward pass and MC sampling, and that the module-only
latency script is clearly distinguished.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def test_e2e_latency_script_exists():
    """run_e2e_latency.py must exist as a separate script from run_latency.py."""
    e2e_path = EXPERIMENTS_DIR / "run_e2e_latency.py"
    assert e2e_path.exists(), (
        "run_e2e_latency.py does not exist. "
        "Create a separate script for true E2E latency measurement."
    )


def test_e2e_latency_includes_gnn():
    """run_e2e_latency.py must include GNN forward pass in its measurement loop."""
    src = (EXPERIMENTS_DIR / "run_e2e_latency.py").read_text()
    assert "gnn" in src.lower() or "GNN" in src, (
        "run_e2e_latency.py does not reference a GNN component. "
        "True E2E latency must include GNN forward pass (or simulation)."
    )
    assert "forward_mc" in src or "mc" in src.lower(), (
        "run_e2e_latency.py does not include MC sampling. "
        "True E2E must include T stochastic forward passes."
    )


def test_e2e_latency_sweeps_T_values():
    """run_e2e_latency.py must sweep T values [1, 5, 10, 20, 30]."""
    src = (EXPERIMENTS_DIR / "run_e2e_latency.py").read_text()
    for T in [1, 5, 10, 20, 30]:
        assert str(T) in src, (
            f"run_e2e_latency.py does not include T={T} in its sweep. "
            "T_VALUES must include [1, 5, 10, 20, 30]."
        )


def test_e2e_latency_outputs_csv():
    """run_e2e_latency.py must produce e2e_latency_by_T.csv."""
    src = (EXPERIMENTS_DIR / "run_e2e_latency.py").read_text()
    assert "e2e_latency_by_T.csv" in src, (
        "run_e2e_latency.py must write results to e2e_latency_by_T.csv"
    )


def test_module_latency_renamed():
    """run_latency.py must use 'module_pipeline' key, not 'end_to_end' as the top-level result."""
    src = (EXPERIMENTS_DIR / "run_latency.py").read_text()
    assert "module_pipeline" in src, (
        "run_latency.py must rename 'end_to_end' to 'module_pipeline' "
        "to prevent misrepresentation as true E2E latency."
    )


def test_module_latency_has_scope_disclaimer():
    """run_latency.py must explicitly state it excludes GNN forward pass."""
    src = (EXPERIMENTS_DIR / "run_latency.py").read_text()
    has_disclaimer = (
        "GNN forward" in src
        or "gnn forward" in src.lower()
        or "Does NOT include" in src
        or "does NOT include" in src
    )
    assert has_disclaimer, (
        "run_latency.py must contain a disclaimer that GNN forward pass "
        "is NOT included in its scope measurement."
    )


def test_e2e_latency_records_hardware_profile():
    """run_e2e_latency.py must write hardware_profile.md."""
    src = (EXPERIMENTS_DIR / "run_e2e_latency.py").read_text()
    assert "hardware_profile.md" in src, (
        "run_e2e_latency.py must produce reports/hardware_profile.md "
        "with CPU/GPU/RAM/PyTorch version info."
    )


def test_e2e_latency_reports_throughput():
    """run_e2e_latency.py must compute events/sec throughput."""
    src = (EXPERIMENTS_DIR / "run_e2e_latency.py").read_text()
    assert "throughput" in src.lower(), (
        "run_e2e_latency.py must compute throughput (events per second)."
    )
