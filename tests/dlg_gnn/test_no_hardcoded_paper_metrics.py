"""
tests/test_no_hardcoded_paper_metrics.py

Round 2 validation: ensure that no paper-facing metric values are
hardcoded in the experiments/ directory.

Checks:
  1. No specific numeric fallback values (0.9369, 0.9906, 0.8264, 0.038) in experiments/
  2. generate_figures.py raises / warns (not silently succeeds) when artifacts are missing
  3. run_multiseed.py output tags results with gnn_source
"""

import ast
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Known hardcoded values that must NOT appear as literal fallback defaults
FORBIDDEN_LITERALS = {
    "0.9369", "0.9906", "0.8264",
    # ECE=0.038 used to be a label in the simulated figure
    "0.038",
    # Old fallback CI bounds
    "0.9318", "0.9419",
    # Old fallback std
    "0.0051", "0.0005", "0.0054",
    # Old fallback recalls
    "0.8699", "0.0113",
}

# Old hardcoded leakage accuracy defaults
FORBIDDEN_LEAK_LITERALS = {"0.6945", "0.7405", "0.9570"}


def _get_literal_strings_and_numbers(source: str):
    """Extract all numeric/string literals from Python source via AST."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            yield str(node.value)


def test_no_hardcoded_metrics_in_generate_figures():
    """generate_figures.py must not contain hardcoded paper metric fallbacks."""
    src = (EXPERIMENTS_DIR / "generate_figures.py").read_text()
    literals = set(_get_literal_strings_and_numbers(src))

    found = FORBIDDEN_LITERALS & literals
    assert not found, (
        f"generate_figures.py contains hardcoded paper metric literals: {found}. "
        "Remove all fallback defaults — use _require_artifact() instead."
    )


def test_no_hardcoded_leak_defaults_in_generate_figures():
    """generate_figures.py must not contain hardcoded leakage accuracy fallbacks."""
    src = (EXPERIMENTS_DIR / "generate_figures.py").read_text()
    literals = set(_get_literal_strings_and_numbers(src))

    found = FORBIDDEN_LEAK_LITERALS & literals
    assert not found, (
        f"generate_figures.py contains hardcoded leakage accuracy defaults: {found}. "
        "All leakage values must come from real attack artifacts."
    )


def test_generate_figures_no_simulated_comment():
    """The old simulated calibration comment must no longer exist."""
    src = (EXPERIMENTS_DIR / "generate_figures.py").read_text()
    assert "Simulated realistic calibrated" not in src, (
        "generate_figures.py still contains the old simulated calibration comment. "
        "Remove the synthetic calibration figure and replace with real data loading."
    )


def test_generate_figures_uses_require_artifact():
    """generate_figures.py must use _require_artifact() guard for paper artifacts."""
    src = (EXPERIMENTS_DIR / "generate_figures.py").read_text()
    assert "_require_artifact" in src, (
        "generate_figures.py does not use _require_artifact(). "
        "Add guards so that missing artifacts cause explicit errors."
    )


def test_run_multiseed_tags_gnn_source():
    """run_multiseed.py must tag output with gnn_source='simulated'."""
    src = (EXPERIMENTS_DIR / "run_multiseed.py").read_text()
    assert "gnn_source" in src, (
        "run_multiseed.py does not include gnn_source in its output. "
        "Add gnn_source='simulated' to the metrics and result dicts."
    )


def test_run_multiseed_saves_raw_predictions():
    """run_multiseed.py must save per-event raw predictions."""
    src = (EXPERIMENTS_DIR / "run_multiseed.py").read_text()
    assert "raw_predictions" in src, (
        "run_multiseed.py does not save raw predictions. "
        "Add code to save event-level scores/labels to results/raw_predictions/."
    )


def test_run_latency_not_called_end_to_end():
    """run_latency.py must not use 'end_to_end' as a meaningful result key."""
    src = (EXPERIMENTS_DIR / "run_latency.py").read_text()
    # The old key "end_to_end" should be deprecated / renamed
    assert "module_pipeline" in src, (
        "run_latency.py still uses 'end_to_end' naming without renaming to 'module_pipeline'. "
        "Round 2 requires the module-only latency to be clearly named."
    )
    assert "does NOT include" in src or "Does NOT include" in src, (
        "run_latency.py must explicitly state what is NOT included in its scope."
    )


@pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_LITERALS))
def test_no_hardcoded_metrics_in_run_all_paper_experiments(forbidden):
    """run_all_paper_experiments.py must not contain paper metric literals as defaults."""
    src_path = EXPERIMENTS_DIR / "run_all_paper_experiments.py"
    if not src_path.exists():
        pytest.skip("run_all_paper_experiments.py not found")
    src = src_path.read_text()
    # Only fail if the literal is used as a fallback default (in default= pattern)
    # Allow if it's in a comment or string comparison
    assert f"default={forbidden}" not in src and f"fallback={forbidden}" not in src, (
        f"run_all_paper_experiments.py contains hardcoded fallback default={forbidden}"
    )
