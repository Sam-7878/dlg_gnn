import pandas as pd
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTROLS_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m2" / "controls"

def test_capacity_controls_raw_completion():
    raw_path = CONTROLS_DIR / "capacity_controls_summary_m2_raw.csv"
    assert raw_path.exists(), f"Missing raw controls summary: {raw_path}"
    
    df_raw = pd.read_csv(raw_path)
    assert len(df_raw) == 45, f"Expected 45 raw runs, found {len(df_raw)}"
    assert (df_raw["status"] == "success").all(), "Not all runs were successful"
    
    # Check datasets
    datasets = set(df_raw["dataset"])
    assert datasets == {"Elliptic", "DGraphFin", "LANL-RedTeam"}
    
    # Check models
    models = set(df_raw["model"])
    assert models == {"DLG-Aug-Zero", "DLG-Aug-Permuted", "DLG-Base-70"}

def test_capacity_controls_aggregated_metrics():
    agg_path = CONTROLS_DIR / "capacity_controls_summary_m2.csv"
    assert agg_path.exists(), f"Missing aggregated summary: {agg_path}"
    
    df = pd.read_csv(agg_path)
    assert len(df) == 9
    
    # Check Elliptic findings
    ell_perm = df[(df["dataset"] == "Elliptic") & (df["model"] == "DLG-Aug-Permuted")].iloc[0]
    ell_zero = df[(df["dataset"] == "Elliptic") & (df["model"] == "DLG-Aug-Zero")].iloc[0]
    ell_base70 = df[(df["dataset"] == "Elliptic") & (df["model"] == "DLG-Base-70")].iloc[0]
    
    # Permuted should drop significantly from authoritative DLG-Aug (~0.2798)
    assert ell_perm["pr_mean"] < 0.15
    assert ell_zero["pr_mean"] < 0.10
    assert ell_base70["pr_mean"] < 0.08
