"""
tests/test_metric_lineage.py

Round 2 validation: verify that the metric computation chain is traceable
from raw predictions → metric functions → aggregated results.

Tests:
  1. run_multiseed.py outputs include per-seed predictions and aggregates
  2. Metric functions produce reproducible results from raw predictions
  3. The lineage report file exists
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


def test_multiseed_output_has_gnn_source_tag():
    """If multiseed_results.json exists, it must contain a gnn_source field."""
    ms_path = REPO_ROOT / "results" / "multiseed" / "multiseed_results.json"
    if not ms_path.exists():
        pytest.skip("multiseed_results.json not yet generated — run experiments first")

    with open(ms_path) as f:
        data = json.load(f)

    assert "gnn_source" in data, (
        "multiseed_results.json missing 'gnn_source' field. "
        "Round 2 requires all paper artifacts to document their data source."
    )


def test_multiseed_output_has_per_seed_metrics():
    """multiseed_results.json must contain per_seed metrics, not just aggregate."""
    ms_path = REPO_ROOT / "results" / "multiseed" / "multiseed_results.json"
    if not ms_path.exists():
        pytest.skip("multiseed_results.json not yet generated")

    with open(ms_path) as f:
        data = json.load(f)

    assert "per_seed" in data, "multiseed_results.json missing 'per_seed' field"
    assert len(data["per_seed"]) >= 1, "per_seed must contain at least one seed result"

    for seed_result in data["per_seed"]:
        assert "auc_pr" in seed_result, f"Seed result missing 'auc_pr': {seed_result}"
        assert "auc_roc" in seed_result, f"Seed result missing 'auc_roc'"


def test_raw_predictions_exist_if_multiseed_exists():
    """If multiseed_results.json exists, at least one raw prediction CSV must exist."""
    ms_path = REPO_ROOT / "results" / "multiseed" / "multiseed_results.json"
    raw_dir = REPO_ROOT / "results" / "raw_predictions"

    if not ms_path.exists():
        pytest.skip("multiseed_results.json not yet generated")

    raw_csvs = list(raw_dir.glob("multiseed_seed*.csv")) if raw_dir.exists() else []
    assert len(raw_csvs) >= 1, (
        "raw_predictions/ directory is empty. "
        "run_multiseed.py must save per-event predictions to results/raw_predictions/."
    )


def test_raw_prediction_csv_has_required_columns():
    """Raw prediction CSV files must contain event_id, label, score columns."""
    raw_dir = REPO_ROOT / "results" / "raw_predictions"
    if not raw_dir.exists():
        pytest.skip("results/raw_predictions/ does not exist yet")

    csvs = list(raw_dir.glob("*.csv"))
    if not csvs:
        pytest.skip("No CSV files in raw_predictions/")

    import pandas as pd
    for csv_path in csvs[:3]:  # Check first 3 files
        df = pd.read_csv(csv_path)
        required = {"label", "score"}
        missing = required - set(df.columns)
        assert not missing, (
            f"{csv_path.name} missing required columns: {missing}. "
            "Raw predictions must contain at least 'label' and 'score'."
        )


def test_metrics_reproducible_from_raw_predictions():
    """AUC-PR computed from raw predictions must match stored aggregate (within tolerance)."""
    ms_path = REPO_ROOT / "results" / "multiseed" / "multiseed_results.json"
    raw_dir = REPO_ROOT / "results" / "raw_predictions"

    if not ms_path.exists() or not raw_dir.exists():
        pytest.skip("Artifacts not available")

    csvs = sorted(raw_dir.glob("multiseed_seed*.csv"))
    if not csvs:
        pytest.skip("No multiseed prediction CSVs available")

    import pandas as pd
    from sklearn.metrics import average_precision_score

    computed_auc_prs = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        if {"label", "score"}.issubset(df.columns):
            try:
                auc = average_precision_score(df["label"].values, df["score"].values)
                computed_auc_prs.append(auc)
            except Exception:
                continue

    if not computed_auc_prs:
        pytest.skip("Could not compute AUC-PR from raw predictions")

    with open(ms_path) as f:
        stored = json.load(f)
    stored_mean = stored.get("aggregate_mean_std", {}).get("auc_pr", {}).get("mean")

    if stored_mean is None:
        pytest.skip("No aggregate AUC-PR in stored results")

    computed_mean = float(np.mean(computed_auc_prs))
    tolerance = 0.005  # allow small floating point differences

    assert abs(computed_mean - stored_mean) < tolerance, (
        f"AUC-PR recomputed from raw predictions ({computed_mean:.4f}) "
        f"does not match stored aggregate ({stored_mean:.4f}). "
        "Check that the same predictions are used in both places."
    )


def test_lineage_report_exists():
    """reports/round_2_metric_lineage.md must exist."""
    report_path = REPO_ROOT / "reports" / "round_2_metric_lineage.md"
    assert report_path.exists(), (
        "reports/round_2_metric_lineage.md does not exist. "
        "Generate the metric lineage report as the first step of Round 2."
    )


def test_lineage_report_mentions_gnn_simulation():
    """The lineage report must explicitly document the GNN simulation."""
    report_path = REPO_ROOT / "reports" / "round_2_metric_lineage.md"
    if not report_path.exists():
        pytest.skip("Lineage report not yet generated")

    content = report_path.read_text()
    assert "simulat" in content.lower(), (
        "round_2_metric_lineage.md must explicitly document that GNN outputs are simulated."
    )
