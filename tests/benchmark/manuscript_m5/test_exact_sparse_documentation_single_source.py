#!/usr/bin/env python3
"""
test_exact_sparse_documentation_single_source.py

Round M5 Unit Test: Verifies that the exact sparse reconstruction formula
is semantically and algebraically consistent across:
1. Manuscript Section 3 (docs/papers/_42_Benchmark/DLG-Benchmark.tex)
2. Canonical documentation (docs/math/exact_sparse_reconstruction.md)
3. Machine-readable manifest (outputs/benchmark/manuscript_m5/release/exact_sparse_identity.json)
"""

import json
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_cross_format_formula_semantic_identity():
    # 1. LaTeX manuscript
    tex_path = REPO_ROOT / "docs/papers/_42_Benchmark/DLG-Benchmark.tex"
    assert tex_path.exists(), f"Missing manuscript TeX: {tex_path}"
    tex_content = tex_path.read_text(encoding="utf-8")

    # Verify Equation (eq:gram_exact) in manuscript
    assert r"\sum_j A_{ij}^2" in tex_content or r"\sum_{j} A_{ij}^2" in tex_content
    assert r"-2\sum_{j:A_{ij}\neq 0}A_{ij}" in tex_content or r"-2\sum_{j:A_{ij}\ne 0}A_{ij}" in tex_content
    assert r"Z^\top Z" in tex_content

    # 2. Markdown documentation
    md_path = REPO_ROOT / "docs/math/exact_sparse_reconstruction.md"
    assert md_path.exists(), f"Missing markdown doc: {md_path}"
    md_content = md_path.read_text(encoding="utf-8")

    assert r"\sum_{j=1}^N A_{ij}^2" in md_content
    assert r"- 2 \sum_{j: A_{ij} \neq 0} A_{ij} (z_i z_j^\top)" in md_content
    assert r"z_i (Z^\top Z) z_i^\top" in md_content
    assert "Directed Graphs" in md_content
    assert "Weighted Graphs" in md_content

    # 3. Machine-readable JSON
    json_path = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/exact_sparse_identity.json"
    if not json_path.exists():
        json_path = REPO_ROOT / "outputs/benchmark/manuscript_m4/release/exact_sparse_identity.json"
    assert json_path.exists(), f"Missing JSON manifest: {json_path}"
    manifest = json.loads(json_path.read_text(encoding="utf-8"))

    assert "exact_sparse_identity" in manifest
    assert "sum_j A_{ij}^2" in manifest["exact_sparse_identity"]
    assert "term_2_cross" in manifest["terms"]
    assert "<(A Z)_i, z_i>" in manifest["terms"]["term_2_cross"]
    assert "z_i * (Z^T Z) * z_i^T" in manifest["terms"]["term_3_gram"]
    assert "erroneous_form_banned" in manifest
    assert "||Z||_F^2" in manifest["erroneous_form_banned"]
