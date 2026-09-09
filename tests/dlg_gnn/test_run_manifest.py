"""
tests/test_run_manifest.py

Round 2 validation: verify that run manifests are generated correctly
and contain all required reproducibility fields.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


def test_run_manifests_directory_created_by_multiseed():
    """run_multiseed.py source must create results/run_manifests/ directory."""
    src_path = REPO_ROOT / "experiments" / "run_multiseed.py"
    src = src_path.read_text()
    assert "run_manifests" in src, (
        "run_multiseed.py must create and write to results/run_manifests/."
    )


def test_manifest_contains_required_fields():
    """Every run manifest must contain the required reproducibility fields."""
    manifest_dir = REPO_ROOT / "results" / "run_manifests"
    if not manifest_dir.exists():
        pytest.skip("run_manifests/ not yet created — run experiments first")

    manifests = list(manifest_dir.glob("*.json"))
    if not manifests:
        pytest.skip("No run manifests yet — run experiments first")

    required_fields = {"git_commit", "seeds", "gnn_source", "config_sha256", "timestamp",
                       "command", "python_version", "torch_version"}

    for mp in manifests:
        with open(mp) as f:
            m = json.load(f)
        missing = required_fields - set(m.keys())
        assert not missing, (
            f"Manifest {mp.name} is missing required fields: {missing}"
        )


def test_manifest_gnn_source_is_labeled():
    """Every manifest must explicitly label the GNN source."""
    manifest_dir = REPO_ROOT / "results" / "run_manifests"
    if not manifest_dir.exists():
        pytest.skip("run_manifests/ not yet created")

    for mp in manifest_dir.glob("*.json"):
        with open(mp) as f:
            m = json.load(f)
        assert "gnn_source" in m, f"Manifest {mp.name} missing 'gnn_source'"
        assert m["gnn_source"] in {"simulated", "real_checkpoint", "unknown"}, (
            f"Manifest {mp.name} has invalid gnn_source: '{m['gnn_source']}'"
        )


def test_manifest_git_commit_is_not_unknown():
    """Git commit in manifests should be real (not 'unknown') if git is available."""
    import subprocess
    try:
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL)
        git_available = True
    except Exception:
        git_available = False

    if not git_available:
        pytest.skip("Git not available")

    manifest_dir = REPO_ROOT / "results" / "run_manifests"
    if not manifest_dir.exists() or not list(manifest_dir.glob("*.json")):
        pytest.skip("No run manifests yet")

    for mp in list(manifest_dir.glob("*.json"))[:3]:
        with open(mp) as f:
            m = json.load(f)
        commit = m.get("git_commit", "unknown")
        assert commit != "unknown", (
            f"Manifest {mp.name} has git_commit='unknown' but git is available. "
            "Fix git commit detection in the experiment script."
        )


def test_dataset_manifest_schema():
    """If dataset_manifest.json exists, it must have the expected schema."""
    dm_path = REPO_ROOT / "results" / "dataset_manifest.json"
    if not dm_path.exists():
        pytest.skip("dataset_manifest.json not yet generated")

    with open(dm_path) as f:
        dm = json.load(f)

    required = {"n_events", "fraud_ratio", "generator_config_hash"}
    missing = required - set(dm.keys())
    assert not missing, (
        f"dataset_manifest.json missing fields: {missing}"
    )
