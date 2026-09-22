import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = REPO_ROOT / "docs" / "papers" / "_42_Benchmark"
TEX_PATH = DOCS_DIR / "DLG-Benchmark.tex"
BIB_PATH = DOCS_DIR / "references.bib"

def test_zero_darpa_theia_occurrences():
    tex_text = TEX_PATH.read_text(encoding="utf-8")
    bib_text = BIB_PATH.read_text(encoding="utf-8")
    
    for term in ["DARPA", "THEIA", "TC-E5", "tc_e5", "darpa_tc"]:
        assert term.lower() not in tex_text.lower(), f"Forbidden term '{term}' found in DLG-Benchmark.tex"
        assert term.lower() not in bib_text.lower(), f"Forbidden term '{term}' found in references.bib"

def test_no_overclaiming_words():
    tex_text = TEX_PATH.read_text(encoding="utf-8")
    
    # Section 5 and Section 7/8 should not use unverified causal or overclaiming phrases
    forbidden_phrases = [
        r'\boversmoothing explains\b',
        r'\bexplains why GADNR\b',
        r'\binduce overfitting\b',
    ]
    for pat in forbidden_phrases:
        match = re.search(pat, tex_text, re.IGNORECASE)
        assert not match, f"Overclaiming pattern '{pat}' found: {match.group(0)}"

def test_bibliography_integrity():
    bib_text = BIB_PATH.read_text(encoding="utf-8")
    tex_text = TEX_PATH.read_text(encoding="utf-8")
    
    # Check invalid survey removed
    assert "gao2024survey" not in bib_text
    assert "gao2024survey" not in tex_text
    
    # Check tang2022revisiting correct authors
    assert "Tang, Jianheng and Li, Jiajin and Gao, Ziqi and Li, Jia" in bib_text
    
    # Check zheng2021generative correct DOI and title
    assert "10.1109/TKDE.2021.3119326" in bib_text
    assert "Generative Pre-Training for Graph Neural Networks" in bib_text

def test_section_roadmap_integrity():
    tex_text = TEX_PATH.read_text(encoding="utf-8")
    # Must list all 9 sections in Introduction roadmap
    roadmap_str = "Section~2 reviews graph anomaly detection, benchmarking, and scalable GNN execution, and clarifies the relation to the preceding DLG-GNN study. Section~3 presents the benchmark framework, DLG variants, exact sparse reconstruction, and support policy. Section~4 describes datasets, provenance, metrics, and the five-seed experimental protocol. Section~5 reports the primary benchmark results, statistical comparisons, topology associations, capacity controls, and LANL external validation. Section~6 evaluates scalability and exact execution support. Section~7 discusses broader implications. Section~8 addresses limitations and threats to validity, and Section~9 concludes the paper."
    assert roadmap_str in tex_text
