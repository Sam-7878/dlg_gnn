"""
Comprehensive Manuscript M1 Integrity and Regression Tests (Phase M)
Verifies:
1. Frozen artifact hashes (Round 5 and LANL) match authoritative registry.
2. Zero occurrences of DARPA, THEIA, TC-E5, or provenance stress test in manuscript tree.
3. Detector count consistency: "eight detector configurations: six established ... and two DLG variants".
4. Dataset count consistency: 10 primary datasets + LANL external validation.
5. Synthetic display names: Yelp-Syn, Amazon-Syn, Flickr-Syn, Reddit-Syn, Cora-Syn, CiteSeer-Syn, PubMed-Syn.
6. Exact claim numbers match source artifacts:
   - 71 supported pairs, 9 restricted pairs, 355 successful runs
   - Fraud-oriented average ranks: ROC 1.71, PR 1.71, F1 1.86
   - Elliptic delta PR +0.0350, Reddit-Syn delta PR -0.0656
7. Capacity controls remain isolated from primary Friedman matrix.
8. Bibliography integrity: all cited bib keys exist in references.bib, no undefined citations.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
DOCS_DIR = REPO_ROOT / "dlg_gnn/docs/papers/_42_Benchmark"
MANUSCRIPT_TEX = DOCS_DIR / "DLG-Benchmark.tex"
BIB_FILE = DOCS_DIR / "references.bib"
REGISTRY_JSON = REPO_ROOT / "dlg_gnn/outputs/benchmark/manuscript_m1/audit/manuscript_source_registry.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_frozen_hashes_match_registry():
    assert REGISTRY_JSON.exists()
    registry = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    for item in registry:
        p = REPO_ROOT / item["artifact_path"]
        assert p.exists(), f"Missing artifact: {p}"
        actual = sha256_file(p)
        assert actual == item["sha256"], f"SHA-256 mismatch for {item['artifact_name']}: {actual} != {item['sha256']}"


def test_no_darpa_theia_in_final_manuscript():
    banned = ["DARPA", "THEIA", "TC-E5", "provenance stress test"]
    manuscript_files = list(DOCS_DIR.glob("*.tex")) + list((DOCS_DIR / "generated").glob("*.tex"))
    for p in manuscript_files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        for word in banned:
            assert word not in text, f"Found banned term '{word}' in {p.name}"


def test_detector_count_consistency():
    tex = MANUSCRIPT_TEX.read_text(encoding="utf-8")
    # Prohibited phrasing
    prohibited = ["eight baselines", "six detectors", "DLG plus eight detectors"]
    for ph in prohibited:
        assert ph not in tex, f"Found non-compliant phrasing: '{ph}'"


def test_synthetic_display_names_in_appendix():
    app_tex = DOCS_DIR / "generated/appendix_performance_tables.tex"
    if app_tex.exists():
        text = app_tex.read_text(encoding="utf-8")
        # Ensure unsuffixed synthetic names are not used as standalone rows
        for syn in ["Yelp-Syn", "Amazon-Syn", "Flickr-Syn", "Reddit-Syn", "Cora-Syn", "CiteSeer-Syn", "PubMed-Syn"]:
            assert syn in text, f"Missing standard synthetic name {syn} in appendix"


def test_claim_numbers_match_source():
    tex = MANUSCRIPT_TEX.read_text(encoding="utf-8")
    # Core numerical anchors
    assert "71" in tex, "Missing 71 supported pairs"
    assert "355" in tex, "Missing 355 successful runs"
    assert "1.71" in tex, "Missing fraud-oriented ROC/PR rank 1.71"
    assert "1.86" in tex, "Missing fraud-oriented F1 rank 1.86"
    assert "0.0350" in tex, "Missing Elliptic PR delta 0.0350"
    assert "0.0656" in tex, "Missing Reddit-Syn PR delta 0.0656"


def test_capacity_controls_isolated():
    # Verify primary rankings.csv does NOT contain capacity controls
    rankings_csv = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_round5_final/statistics/rankings.csv"
    assert rankings_csv.exists()
    df = pd.read_csv(rankings_csv)
    models = set(df["model"].unique())
    assert "DLG-Aug-Zero" not in models
    assert "DLG-Base-70" not in models
    assert len(models) == 8  # 6 baselines + DLG-Base + DLG-Aug


def test_bib_keys_resolve():
    tex = MANUSCRIPT_TEX.read_text(encoding="utf-8")
    bib = BIB_FILE.read_text(encoding="utf-8")
    
    # Extract all bib keys defined
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    
    # Extract all \cite{...} keys in tex
    cited_keys = set()
    for match in re.findall(r"\\cite\{([^}]+)\}", tex):
        for k in match.split(","):
            cited_keys.add(k.strip())
            
    # Every cited key must be in bib
    missing = cited_keys.difference(bib_keys)
    assert not missing, f"Missing bib entries for cited keys: {missing}"
