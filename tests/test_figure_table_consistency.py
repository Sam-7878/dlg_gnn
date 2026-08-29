"""
tests/test_figure_table_consistency.py

Round 2 validation: verify that all summary CSVs, LaTeX tables,
and raw experiment JSON artifacts have consistent metrics.

Checks:
  1. main_results.csv values match multiseed_results.json aggregate values
  2. LaTeX tables match corresponding CSVs (when generated)
  3. No NaN values in published summary tables
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def test_main_results_matches_multiseed_json():
    """Check that main_results.csv accurately mirrors multiseed_results.json."""
    json_path = RESULTS_DIR / "multiseed" / "multiseed_results.json"
    csv_path = RESULTS_DIR / "main_results.csv"

    if not json_path.exists() or not csv_path.exists():
        pytest.skip("multiseed results or main_results.csv not found")

    with open(json_path) as f:
        data = json.load(f)
    df = pd.read_csv(csv_path)

    agg = data.get("aggregate_mean_std", {})
    if "auc_pr" in agg and "AUC-PR_mean" in df.columns:
        expected = float(agg["auc_pr"]["mean"])
        actual = float(df["AUC-PR_mean"].iloc[0])
        assert abs(expected - actual) < 1e-4, (
            f"AUC-PR mismatch: CSV has {actual}, JSON has {expected}"
        )

    if "auc_roc" in agg and "AUC-ROC_mean" in df.columns:
        expected = float(agg["auc_roc"]["mean"])
        actual = float(df["AUC-ROC_mean"].iloc[0])
        assert abs(expected - actual) < 1e-4, (
            f"AUC-ROC mismatch: CSV has {actual}, JSON has {expected}"
        )


def test_no_unintended_nans_in_summary_csvs():
    """Summary CSVs should not have all-NaN columns or corrupted rows."""
    for fname in ["main_results.csv", "context_baselines.csv", "e2e_latency_by_T.csv"]:
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        assert len(df) > 0, f"{fname} is empty"
        # Ensure at least one metric column is not all NaN
        non_obj_cols = df.select_dtypes(include=[np.number]).columns
        if len(non_obj_cols) > 0:
            assert df[non_obj_cols].notna().any().all(), (
                f"{fname} has columns that are entirely NaN"
            )


def test_latex_table_consistency_if_exists():
    """Verify that generated LaTeX tables contain the same primary metric values as CSVs."""
    tbl_dir = RESULTS_DIR / "tables"
    if not tbl_dir.exists():
        pytest.skip("results/tables/ not generated yet")

    main_tex = tbl_dir / "main_results.tex"
    main_csv = RESULTS_DIR / "main_results.csv"

    if main_tex.exists() and main_csv.exists():
        tex_content = main_tex.read_text(encoding="utf-8")
        df_main = pd.read_csv(main_csv)
        if "AUC-PR_mean" in df_main.columns:
            val_str = f"{df_main['AUC-PR_mean'].iloc[0]:.4f}"
            # Check either exact representation or rounded
            val_float = float(df_main['AUC-PR_mean'].iloc[0])
            found = (val_str in tex_content) or (f"{val_float:.3f}" in tex_content) or (f"{val_float}" in tex_content)
            assert found, (
                f"LaTeX table main_results.tex does not reflect CSV AUC-PR value {val_str}"
            )
