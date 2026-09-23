#!/usr/bin/env python3
"""
test_preprint_mdpi_abstract_identity.py

Round P3 Gate: Verifies that the compressed abstract is 100% scientifically
identical across master, Preprint, and MDPI manuscripts, and preserves all
mandatory empirical elements:
  - 8 detector configurations (6 baselines + DLG-Base & DLG-Aug)
  - Frozen 10-dataset primary suite and LANL-RedTeam external validation
  - 5 seeds and 355 successful runs (71 of 80 supported pairs)
  - Fraud-oriented common subset rankings (1.71 ROC-AUC, 1.71 PR-AUC, 1.86 F1)
  - Elliptic (+0.0350) and Reddit-Syn (-0.0656) augmentation deltas
  - Bounded conclusion: local augmentation is conditional rather than universal
"""

from pathlib import Path
import re
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MASTER_TEX = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "DLG-Benchmark.tex"
PREPRINT_TEX = REPO_ROOT / "publication" / "benchmark" / "preprints" / "DLG-Benchmark-Preprint.tex"
MDPI_TEX = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "DLG-Benchmark.tex"


def extract_clean_abstract(tex_path: Path) -> str:
    content = tex_path.read_text(encoding="utf-8")
    m = re.search(r"\\abstract\{([\s\S]*?)\}\s*\\keyword\{", content)
    if m:
        raw = m.group(1)
    else:
        m2 = re.search(r"\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}", content)
        if m2:
            raw = m2.group(1)
            raw = re.sub(r"\\vspace\{[^}]+\}", "", raw)
            raw = re.sub(r"\\noindent\\textbf\{Keywords:\}.*", "", raw)
        else:
            raise ValueError(f"Could not extract abstract from {tex_path.name}")
    # Normalize whitespace
    return re.sub(r"\s+", " ", raw).strip()


def test_abstract_exact_identity():
    mast_abs = extract_clean_abstract(MASTER_TEX)
    prep_abs = extract_clean_abstract(PREPRINT_TEX)
    mdpi_abs = extract_clean_abstract(MDPI_TEX)

    assert prep_abs == mdpi_abs, "Preprint and MDPI abstracts differ!"
    assert prep_abs == mast_abs, "Preprint and Master abstracts differ!"


def test_abstract_contains_mandatory_elements():
    abs_text = extract_clean_abstract(MASTER_TEX)

    # Core counts
    assert "eight detector configurations" in abs_text
    assert "ten-dataset primary suite" in abs_text or "10-dataset" in abs_text
    assert "LANL-RedTeam" in abs_text
    assert "five seeds" in abs_text
    assert "71 of 80" in abs_text or "71/80" in abs_text
    assert "355 successful runs" in abs_text

    # Baselines & models
    assert "DLG-Base" in abs_text
    assert "DLG-Aug" in abs_text

    # Fraud rank
    assert "1.71" in abs_text
    assert "1.86" in abs_text

    # Key deltas
    assert "0.0350" in abs_text
    assert "0.0656" in abs_text
    assert "Elliptic" in abs_text
    assert "Reddit-Syn" in abs_text

    # Nuanced / conditional conclusion
    assert "conditional rather than universal" in abs_text
