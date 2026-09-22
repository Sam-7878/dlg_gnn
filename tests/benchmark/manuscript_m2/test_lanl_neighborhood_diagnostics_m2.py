import json
import pandas as pd
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LANL_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m2" / "lanl"
TEX_PATH = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "DLG-Benchmark.tex"

def test_lanl_diagnostics_csv_and_json():
    csv_path = LANL_DIR / "lanl_neighborhood_diagnostics_m2.csv"
    json_path = LANL_DIR / "lanl_neighborhood_diagnostics_m2.json"
    assert csv_path.exists()
    assert json_path.exists()
    
    df = pd.read_csv(csv_path)
    assert len(df) == 7
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["num_nodes"] == 16694
    assert data["num_edges"] == 323897
    assert data["num_features"] == 13
    assert data["num_positives"] == 301
    assert data["num_negatives"] == 16393
    
    in_deg = data["metrics"]["in_degree"]
    assert in_deg["pos_median"] == 9.0
    assert in_deg["neg_median"] == 4.0
    assert round(in_deg["cliffs_delta"], 3) == 0.819

def test_manuscript_section_5_7_numbers_match_csv():
    tex_text = TEX_PATH.read_text(encoding="utf-8")
    
    # Must contain canonical numbers
    assert "median 9.0, IQR 15.0 vs.\\ normal median 4.0, IQR 4.0" in tex_text
    assert "Cliff's $\\delta = +0.819$" in tex_text
    assert "median 31.0, IQR 24.0 vs.\\ 21.0, IQR 13.0" in tex_text
    assert "Cliff's $\\delta = +0.630$" in tex_text
    assert "median 33.0, IQR 26.0 vs.\\ 23.0, IQR 14.0" in tex_text
    assert "Cliff's $\\delta = +0.633$" in tex_text
    assert "155,795 edges connect anomalous to normal nodes" in tex_text
    assert "4,201 anomaly--anomaly edges" in tex_text
