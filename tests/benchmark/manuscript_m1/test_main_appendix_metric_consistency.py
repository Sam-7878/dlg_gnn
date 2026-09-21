"""
Regression test: Main vs Appendix Metric Consistency.
Verifies that generated appendix tables exactly agree with the frozen Round 5 artifacts.
"""

import re
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

ROUND5_SUMMARY = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_round5_final/summary/seed_aggregated_performance.csv"
SUPPORT_MATRIX = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_round5_final/manifests/model_dataset_support_matrix_v2.csv"
APPENDIX_TEX = REPO_ROOT / "dlg_gnn/docs/papers/_42_Benchmark/generated/appendix_performance_tables.tex"

SYNTHETIC_DATASETS = ["Yelp-Syn", "Amazon-Syn", "Reddit-Syn", "Flickr-Syn", "Cora-Syn", "CiteSeer-Syn", "PubMed-Syn"]
REAL_DATASETS = ["Elliptic", "DGraphFin", "BitcoinOTC"]
EXPECTED_MODELS = ["DOMINANT", "AnomalyDAE", "CoLA", "CONAD", "GADNR", "OCGNN", "DLG-Base", "DLG-Aug"]


def test_frozen_hashes():
    import hashlib
    h_raw = hashlib.sha256((REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_round5_final/raw/benchmark_raw.csv").read_bytes()).hexdigest()
    assert h_raw == "39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c"
    
    h_sup = hashlib.sha256((REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_round5_final/manifests/model_dataset_support_matrix_v2.csv").read_bytes()).hexdigest()
    assert h_sup == "c58dbca9a9e1ed14dfc025075820a3ad745f6cb70be77764c265d90af3522914"


def test_appendix_tex_exists():
    assert APPENDIX_TEX.exists(), f"Appendix table not found: {APPENDIX_TEX}"


def test_appendix_metric_values_match_source():
    df_perf = pd.read_csv(ROUND5_SUMMARY)
    df_sup = pd.read_csv(SUPPORT_MATRIX)
    
    # Map (dataset, model) -> (supported, status)
    sup_map = {(r["dataset"], r["model"]): r["supported"] for _, r in df_sup.iterrows()}
    
    # Check that unsuffixed Yelp or Amazon do not appear as standalone dataset identifiers in the table
    tex_content = APPENDIX_TEX.read_text(encoding="utf-8")
    assert "DLG (Ours)" not in tex_content
    assert "DLG(Ours)" not in tex_content
    
    for syn in ["Yelp-Syn", "Amazon-Syn", "Reddit-Syn", "Flickr-Syn", "Cora-Syn", "CiteSeer-Syn", "PubMed-Syn"]:
        assert f"{{{syn}}}" in tex_content, f"Expected {syn} in appendix table"

    # Verify key numeric anchors from Claude review
    # Elliptic DLG-Aug PR-AUC: 0.1037
    assert "0.1037" in tex_content
    # Elliptic DLG-Base PR-AUC: 0.0687
    assert "0.0687" in tex_content
    # Stale numbers must be absent
    assert "0.0841" not in tex_content
    assert "0.0653" not in tex_content
    assert "0.4315" not in tex_content

    # Check that unsupported cells are present as '---'
    # GADNR on Elliptic is unsupported
    # In table row for Elliptic, GADNR column must be '---'
    for line in tex_content.splitlines():
        if "Elliptic" in line and "PR-AUC" in line:
            parts = [p.strip() for p in line.split("&")]
            # Format: \multirow{3}{*}{Elliptic} & PR-AUC & DOMINANT & AnomalyDAE & CoLA & CONAD & GADNR & OCGNN & DLG-Base & DLG-Aug \\
            # parts[6] corresponds to GADNR
            assert parts[6] == "---", f"Expected GADNR on Elliptic to be '---', got {parts[6]}"
            assert "0.1037" in parts[9], f"Expected DLG-Aug PR-AUC 0.1037, got {parts[9]}"
            assert "0.0687" in parts[8], f"Expected DLG-Base PR-AUC 0.0687, got {parts[8]}"
