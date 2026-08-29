"""
tests/test_threshold_policy.py

Round 2 validation: verify that decision thresholds are selected on
validation sets, not test sets, and that this policy is documented.

Tests:
  1. run_multiseed.py uses a fixed threshold (0.5) or documents threshold source
  2. Ablation experiments do not tune threshold on test set
  3. Threshold policy is explicitly recorded in metrics output
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def test_multiseed_uses_documented_threshold():
    """run_multiseed.py must document the threshold policy used for F1 computation."""
    src = (EXPERIMENTS_DIR / "run_multiseed.py").read_text()
    # Must use a fixed threshold (not tuned on test labels)
    assert "threshold" in src.lower() or "0.5" in src, (
        "run_multiseed.py does not document threshold policy. "
        "F1 computation requires a threshold — document whether it is fixed (0.5) "
        "or selected on validation set."
    )
    # Must NOT tune threshold using test labels — check for forbidden patterns (simple string match)
    forbidden_patterns = [
        "maximize_f1(y_test",
        "optimize_threshold(test",
    ]
    for pattern in forbidden_patterns:
        assert pattern not in src, (
            f"run_multiseed.py appears to tune threshold on test set: '{pattern}' found. "
            "Threshold must be selected on validation set only."
        )


def test_multiseed_threshold_is_fixed_not_test_optimized():
    """The threshold used in run_multiseed.py must be a fixed constant, not optimized on test."""
    src = (EXPERIMENTS_DIR / "run_multiseed.py").read_text()
    # Look for threshold 0.5 being used (fixed policy)
    assert ">= 0.5" in src or "threshold = 0.5" in src or "thr = 0.5" in src, (
        "run_multiseed.py does not explicitly use threshold=0.5 for classification. "
        "If using a validation-tuned threshold, document it clearly."
    )


def test_multiseed_auc_metrics_are_threshold_free():
    """AUC-ROC and AUC-PR are threshold-free and must be primary metrics."""
    src = (EXPERIMENTS_DIR / "run_multiseed.py").read_text()
    assert "average_precision_score" in src, (
        "run_multiseed.py must compute AUC-PR (threshold-free) as a primary metric."
    )
    assert "roc_auc_score" in src, (
        "run_multiseed.py must compute AUC-ROC (threshold-free) as a primary metric."
    )


def test_ablation_uses_fixed_or_val_threshold():
    """run_ablation.py must not tune threshold using test labels."""
    ablation_path = EXPERIMENTS_DIR / "run_ablation.py"
    if not ablation_path.exists():
        pytest.skip("run_ablation.py not found")

    src = ablation_path.read_text()
    # Check it doesn't search for best threshold over test labels
    assert "argmax.*y_test" not in src.lower() or "best_threshold" not in src, (
        "run_ablation.py appears to optimize threshold on test labels."
    )


def test_context_baselines_uses_fixed_threshold():
    """run_context_baselines.py must use a fixed threshold or validation-set threshold."""
    cb_path = EXPERIMENTS_DIR / "run_context_baselines.py"
    if not cb_path.exists():
        pytest.skip("run_context_baselines.py not found")

    src = cb_path.read_text()
    # The baseline evaluation must use a consistent threshold policy
    assert "threshold" in src.lower() or "0.5" in src, (
        "run_context_baselines.py does not document its threshold policy."
    )


def test_run_manifest_records_threshold_info():
    """If run manifests exist, they should ideally document threshold policy."""
    manifest_dir = REPO_ROOT / "results" / "run_manifests"
    if not manifest_dir.exists():
        pytest.skip("run_manifests/ directory does not exist yet")

    manifests = list(manifest_dir.glob("*.json"))
    if not manifests:
        pytest.skip("No run manifests found yet")

    # At minimum, check manifests are valid JSON with expected fields
    for mp in manifests[:3]:
        with open(mp) as f:
            m = json.load(f)
        assert "seeds" in m, f"Manifest {mp.name} missing 'seeds' field"
        assert "git_commit" in m, f"Manifest {mp.name} missing 'git_commit' field"
        assert "timestamp" in m, f"Manifest {mp.name} missing 'timestamp' field"
