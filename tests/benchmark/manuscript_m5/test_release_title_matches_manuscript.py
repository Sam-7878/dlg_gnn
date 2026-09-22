#!/usr/bin/env python3
r"""
test_release_title_matches_manuscript.py

Round M5 Unit Test: Verifies that the official paper title matches exactly
across all public-facing publication and packaging files:
1. DLG-Benchmark.tex (\Title{...})
2. pyproject.toml (description)
3. CITATION.cff (title)
4. release_metadata.json (paper_title)
"""

import json
from pathlib import Path
import tomllib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_TITLE = "A Reproducible and Scalability-Aware Benchmark for Graph Anomaly Detection with Decoupled Local-to-Global GNNs"


def test_manuscript_title():
    tex_path = REPO_ROOT / "docs/papers/_42_Benchmark/DLG-Benchmark.tex"
    assert tex_path.exists()
    content = tex_path.read_text(encoding="utf-8")
    assert f"\\Title{{{EXPECTED_TITLE}}}" in content
    assert f"\\TitleCitation{{{EXPECTED_TITLE}}}" in content


def test_pyproject_title():
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.exists()
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    assert data["project"]["description"] == EXPECTED_TITLE


def test_citation_cff_title():
    cff_path = REPO_ROOT / "CITATION.cff"
    assert cff_path.exists()
    data = yaml.safe_load(cff_path.read_text(encoding="utf-8"))
    assert data["title"] == EXPECTED_TITLE
    assert data["preferred-citation"]["title"] == EXPECTED_TITLE


def test_release_metadata_title():
    meta_path = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/release_metadata.json"
    assert meta_path.exists()
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["paper_title"] == EXPECTED_TITLE
